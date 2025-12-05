from AlgorithmImports import *
from datetime import datetime, timedelta, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Small Actor–Critic Network (shared pattern for VIX + ES)
class ActorCriticNet(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.nn.functional.gelu(self.ln1(self.fc1(x)))
        x = torch.nn.functional.gelu(self.ln2(self.fc2(x)))
        x = torch.nn.functional.gelu(self.ln3(self.fc3(x)))
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


# ============================================================
# Main Algorithm
# ============================================================
class FormalVioletDonkey(QCAlgorithm):

    def Initialize(self):
        self.MODE = "TRAIN"
        # === Basic backtest setup ===
        if self.MODE == "TRAIN":
            self.SetStartDate(2023, 1, 1)
            self.SetEndDate(2024, 5, 1)
        else:
            self.SetStartDate(2024, 5, 1)
            self.SetEndDate(2024, 8, 1)
        self.SetCash(1000000)
        self.SetWarmup(30, Resolution.Daily)

        # === RL training vs evaluation split ===
        # Before this date: online learning (actor–critic updates)
        # After this date: frozen policies (evaluation only)
        self.training_end_date = datetime(2025, 1, 1)

        # === VIX futures (VX) ===
        self.vix_future = self.AddFuture(Futures.Indices.VIX, Resolution.Minute)
        self.vix_future.SetFilter(timedelta(0), timedelta(180))
        self.vix_future_symbol = self.vix_future.Symbol

        # === ES futures (E-mini S&P 500) ===
        self.es_future = self.AddFuture(Futures.Indices.SP_500_E_MINI, Resolution.Minute)
        self.es_future.SetFilter(timedelta(0), timedelta(180))
        self.es_future_symbol = self.es_future.Symbol

        # === Spot VIX & VVIX indices ===
        self.vix_index = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol

        try:
            self.vvix_index = self.AddData(CBOE, "VVIX", Resolution.Daily).Symbol
        except:
            # if VVIX not available, fallback to VIX as a proxy
            self.vvix_index = self.vix_index
        try:
            self.vix_contango = self.AddData(VIXCentralContango, "VX", Resolution.Daily).Symbol
        except:
            self.vix_contango = None

        # --- cached daily regime values ---
        self.last_spot_vix = None
        self.last_vvix = None
        self.last_contango_roll = None

        # --- storage for roll (for ΔRoll) ---
        self.prev_roll = None

        # === Multi-agent system (RL VIX + RL ES + rule-based risk manager) ===
        self.multi_agent = MultiAgentSystem(self)


        if self.MODE == "TEST":
            self.multi_agent.trader.LoadModel("vix_rl_model")
            self.multi_agent.hedger.LoadModel("es_rl_model")

            # Freeze parameters
            for p in self.multi_agent.trader.net.parameters():
                p.requires_grad = False
            for p in self.multi_agent.hedger.net.parameters():
                p.requires_grad = False
    # OnData: pass slices to agents + update regime caches + maybe decide
    
    def OnData(self, data: Slice):
        # --- update cached daily values (VIX/VVIX/contango) ---
        if data.ContainsKey(self.vix_index):
            try:
                self.last_spot_vix = float(data[self.vix_index].Close)
            except:
                pass

        if self.vvix_index is not None and data.ContainsKey(self.vvix_index):
            try:
                self.last_vvix = float(data[self.vvix_index].Close)
            except:
                pass

        if self.vix_contango is not None and data.ContainsKey(self.vix_contango):
            contango_point = data[self.vix_contango]
            if hasattr(contango_point, "ContangoF2F1"):
                self.last_contango_roll = float(contango_point.ContangoF2F1)

        # --- let multi-agent system collect intraday state ---
        self.multi_agent.OnData(data)

        # --- ask multi-agent system to possibly make a decision (intra-day) ---
        self.multi_agent.MaybeDecide()
    # End-of-algorithm reporting
    def OnEndOfAlgorithm(self):
        # Let multi-agent system print Sharpe etc.
        self.multi_agent.OnEnd()

# Multi-Agent System Coordinator
class MultiAgentSystem:
    """
    Wires together:
      - VIXTraderAgent  (RL Actor–Critic Agent 1)
      - ESHedgeAgent    (RL Actor–Critic Agent 2)
      - RiskManagerAgent (Rule-based meta-agent)

    Collects state intraday, and decides at most once per decision_interval.
    """

    def __init__(self, algorithm: QCAlgorithm):
        self.algorithm = algorithm

        # === RL agents ===
        self.trader = VIXTraderAgent(algorithm)
        self.hedger = ESHedgeAgent(algorithm, self.trader)

        # Risk manager remains rule-based, but sees RL agents' stats
        self.risk_manager = RiskManagerAgent(algorithm, self.trader, self.hedger)

        # For performance stats / Sharpe at end
        self.returns = deque(maxlen=500)
        self.last_equity = None
        self.last_return_date = None

        # Intra-day decision throttling
        self.last_decision_time = None
        self.decision_interval = timedelta(minutes=60)  # change to 15/5 if you want even more action
    # Intraday: collect state for all agents
    def OnData(self, data: Slice):
        """
        Called every bar from the main algorithm.

        - RL agents: update their current_obs / has_state
        - Risk manager: update its rolling Sharpe/drawdown state
        """
        self.trader.CollectState(data)
        self.hedger.CollectState(data)
        self.risk_manager.CollectState(data)
    # Intra-day decision point, throttled by decision_interval
    
    def MaybeDecide(self):
        algo = self.algorithm
        if algo.IsWarmingUp:
            return

        now = algo.Time
        if self.last_decision_time is not None:
            if now - self.last_decision_time < self.decision_interval:
                return
        self.last_decision_time = now

        # Is this step in training phase or evaluation phase?
        train_phase = (algo.MODE == "TRAIN")


        # Current portfolio equity
        current_equity = algo.Portfolio.TotalPortfolioValue

        # 1) RL reward updates (equity delta since last action)
        self.trader.UpdateFromReward(current_equity, train_phase)
        self.hedger.UpdateFromReward(current_equity, train_phase)

        # 2) Update daily return for risk stats
        self._update_daily_return()

        # Need at least some history before risk manager can act
        if not self.trader.has_state:
            return

        # Allow risk manager to default to neutral allocation
        if not self.risk_manager.has_state:
            alloc_trader, alloc_hedger = 0.7, 0.3
        else:
            alloc_action = self.risk_manager.DecideAction()
            alloc_trader, alloc_hedger = self.risk_manager.DecodeAllocationAction(alloc_action)

        self.trader.SetAllocation(alloc_trader)
        self.hedger.SetAllocation(alloc_hedger)

        # 3) Risk manager: choose allocation (0..1 for each agent)
        alloc_action = self.risk_manager.DecideAction()
        alloc_trader, alloc_hedger = self.risk_manager.DecodeAllocationAction(alloc_action)
        self.trader.SetAllocation(alloc_trader)
        self.hedger.SetAllocation(alloc_hedger)

        # 4) RL VIX trader decision (Actor–Critic)
        self.trader.DecideAndAct(train_phase)

        # 5) RL ES hedge decision (after VX trade)
        self.hedger.RefreshAfterTrader()
        self.hedger.DecideAndAct(train_phase)
    # Daily return tracking for Sharpe / drawdown
    def _update_daily_return(self):
        algo = self.algorithm
        equity = algo.Portfolio.TotalPortfolioValue
        d = algo.Time.date()

        if self.last_equity is None:
            self.last_equity = equity
            self.last_return_date = d
            return

        # Only log one daily return per day
        if self.last_return_date == d:
            return

        if self.last_equity > 0:
            ret = (equity - self.last_equity) / self.last_equity
            self.returns.append(ret)
            self.risk_manager.OnNewReturn(ret)

        self.last_equity = equity
        self.last_return_date = d
    # End-of-algorithm reporting
    def OnEnd(self):
        algo = self.algorithm
        if len(self.returns) > 2:
            arr = np.array(self.returns)
            sharpe = float(np.mean(arr) / (np.std(arr) + 1e-6))
            algo.Debug(f"Approx daily Sharpe (unannualized): {sharpe:.3f}")
        
        if self.algorithm.Time < self.algorithm.training_end_date:
            return  # don't save mid-training

        # At the end of training phase
        self.trader.SaveModel("vix_rl_model")
        self.hedger.SaveModel("es_rl_model")
# Agent 1: VIX Trader (RL Actor–Critic)
class VIXTraderAgent:
    """
    RL environment for trading front VIX futures.
    Observation = feature vector (9 dims)
      [front, second, roll, slope, delta_roll, first30, basis, rv_plus, rv_minus]

    Action = index into POSITION_FACTORS (-1.0, -0.5, 0.0, 0.5, 1.0)

    Reward (for learning) = change in total portfolio equity since last action,
    scaled, plus value baseline (actor–critic).
    """
    POSITION_FACTORS = [-1.0, -0.5, 0.0, 0.5, 1.0]

    def __init__(self, algorithm: QCAlgorithm):
        self.algorithm = algorithm

        # risk-manager-controlled notional scaling
        self.allocation = 1.0

        self.future_chain_symbol = algorithm.vix_future_symbol
        self.front_contract_symbol = None
        self.prev_front_symbol = None

        self.current_obs = None
        self.has_state = False

        # Intraday tracking for RV+/RV-, first 30m return
        self.current_day = None
        self.day_open_price = None
        self.first_30m_return = 0.0
        self.intraday_returns = []
        self.last_price = None

        self.last_rv_plus = 0.0
        self.last_factor = 0.0

        # === RL-specific ===
        self.obs_dim = 9
        self.num_actions = len(self.POSITION_FACTORS)
        self.device = torch.device("cpu")

        self.net = ActorCriticNet(self.obs_dim, self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        # Last transition info for actor–critic update
        self.last_obs_tensor = None
        self.last_action_tensor = None
        self.last_equity = None

        # Reward scaling (equity deltas)
        self.reward_scale = 10000.0
        self.entropy_coef = 0.001
        self.critic_coef = 0.5

    def SaveModel(self, key: str):
        import pickle
        data = pickle.dumps(self.net.state_dict())
        self.algorithm.ObjectStore.SaveBytes(key, data)
        self.algorithm.Debug(f"Saved model to ObjectStore as {key}")
        if self.algorithm.ObjectStore.ContainsKey(key):
            data = self.algorithm.ObjectStore.ReadBytes(key)
            self.algorithm.Debug(f"Model {key} saved. Size = {len(data)} bytes")
        else:
            self.algorithm.Debug(f"Model {key} NOT FOUND")

    def LoadModel(self, key: str):
        import pickle
        if self.algorithm.ObjectStore.ContainsKey(key):
            data = self.algorithm.ObjectStore.ReadBytes(key)
            self.algorithm.Debug(f"Model {key} saved. Size = {len(data)} bytes")
        else:
            self.algorithm.Debug(f"Model {key} NOT FOUND")
        data = self.algorithm.ObjectStore.ReadBytes(key)
        state_dict = pickle.loads(data)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.algorithm.Debug(f"Loaded model from ObjectStore: {key}")

    # Risk manager hook
    
    def SetAllocation(self, allocation: float):
        self.allocation = max(0.0, min(1.0, allocation))
    # State building
    def _select_front_second(self, contracts):
        algo = self.algorithm
        now = algo.Time
        filtered = [c for c in contracts if c.Expiry.date() >= (now + timedelta(days=7)).date()]
        if len(filtered) >= 2:
            return filtered[0], filtered[1]
        return contracts[0], contracts[1]

    def _update_intraday_features(self, front_price: float):
        algo = self.algorithm
        now_date = algo.Time.date()

        if self.current_day is None or now_date != self.current_day:
            self.current_day = now_date
            self.day_open_price = front_price
            self.first_30m_return = 0.0
            self.intraday_returns = []
            self.last_price = front_price

        if self.last_price is not None and self.last_price > 0:
            r = (front_price - self.last_price) / self.last_price
            self.intraday_returns.append(r)
        self.last_price = front_price

        if self.day_open_price is not None:
            if (self.first_30m_return == 0.0 and algo.Time.time() >= time(10, 0)):
                self.first_30m_return = (front_price - self.day_open_price) / self.day_open_price

    def CollectState(self, data: Slice):
        algo = self.algorithm

        chains = data.FutureChains
        if self.future_chain_symbol not in chains:
            self.current_obs = None
            self.has_state = False
            return

        chain = chains[self.future_chain_symbol]
        contracts = sorted(chain, key=lambda x: x.Expiry)
        if len(contracts) < 2:
            self.current_obs = None
            self.has_state = False
            return

        front, second = self._select_front_second(contracts)
        front_price = float(front.LastPrice)
        second_price = float(second.LastPrice)

        if front_price <= 0 or second_price <= 0:
            self.current_obs = None
            self.has_state = False
            return

        self.front_contract_symbol = front.Symbol

        # Term structure
        slope = second_price - front_price
        roll = slope
        if algo.last_contango_roll is not None:
            roll = algo.last_contango_roll

        delta_roll = 0.0
        if algo.prev_roll is not None:
            delta_roll = roll - algo.prev_roll
        algo.prev_roll = roll

        # Basis from cached spot VIX
        basis = 0.0
        if algo.last_spot_vix is not None:
            basis = front_price - algo.last_spot_vix

        # Intraday RV
        self._update_intraday_features(front_price)

        intr = np.array(self.intraday_returns) if self.intraday_returns else np.array([])
        if intr.size > 0:
            rv_plus = float(np.sum(np.square(intr[intr > 0])))
            rv_minus = float(np.sum(np.square(intr[intr < 0])))
        else:
            rv_plus = 0.0
            rv_minus = 0.0

        self.last_rv_plus = rv_plus
        first30 = float(self.first_30m_return)

        self.current_obs = np.array([
            front_price, second_price,
            roll, slope, delta_roll,
            first30, basis,
            rv_plus, rv_minus
        ], dtype=np.float32)

        self.has_state = True
    def UpdateFromReward(self, current_equity: float, train_phase: bool):
        """
        Use the change in equity since last action as reward for that action,
        and perform a one-step actor–critic update.

        Called from MaybeDecide() *before* choosing the next action.
        """
        if not train_phase:
            self.last_equity = current_equity
            return   # <- prevents training in TEST
        if (not train_phase or
            self.last_obs_tensor is None or
            self.last_action_tensor is None or
            self.last_equity is None):
            self.last_equity = current_equity
            return

        # reward = equity change since last action (scaled)
        reward = (current_equity - self.last_equity) / self.reward_scale
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)

        logits, value = self.net(self.last_obs_tensor)      # value shape: (1,)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(self.last_action_tensor)    # (1,)

        # Advantage = r - V(s) (gamma=0 bandit-style actor–critic)
        advantage = reward_t - value.detach()

        # Actor loss: maximize log(pi(a|s)) * advantage
        actor_loss = -logprob * advantage

        # Critic loss: MSE between V(s) and reward
        critic_loss = F.mse_loss(value, reward_t)

        # Entropy bonus for exploration
        entropy = dist.entropy().mean()

        loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        # Update last_equity to current for the next step's reward
        self.last_equity = current_equity
    def DecideAndAct(self, train_phase: bool):
        algo = self.algorithm

        if not self.has_state or self.current_obs is None:
            return

        # Build observation tensor
        obs = self.current_obs
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Forward through actor–critic network
        logits, value = self.net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)

        if train_phase:
            # Sample action for exploration
            action_tensor = dist.sample()
        else:
            # Greedy action at evaluation time
            action_tensor = torch.argmax(logits, dim=-1)

        action_index = int(action_tensor.item())

        # Save minimal info for next reward update
        self.last_obs_tensor = obs_t          # (1, obs_dim)
        self.last_action_tensor = action_tensor  # (1,)
        self.last_equity = algo.Portfolio.TotalPortfolioValue

        # Execute the trade in VX futures
        self._execute_action(action_index)
    # Trade execution logic
    def _execute_action(self, action_index: int):
        algo = self.algorithm
        if self.front_contract_symbol is None:
            return

        # Roll handling: if front contract changed, liquidate old VX positions
        if self.prev_front_symbol is not None and self.prev_front_symbol != self.front_contract_symbol:
            old_qty = algo.Portfolio[self.prev_front_symbol].Quantity
            if old_qty != 0:
                algo.MarketOrder(self.prev_front_symbol, -old_qty)

        self.prev_front_symbol = self.front_contract_symbol

        factor = self.POSITION_FACTORS[int(action_index)]
        self.last_factor = factor

        symbol = self.front_contract_symbol
        price = algo.Securities[symbol].Price
        if price <= 0:
            return

        equity = algo.Portfolio.TotalPortfolioValue
        max_notional_fraction = 0.08  # increase if you want bigger swings
        target_notional = equity * max_notional_fraction * self.allocation * factor
        contract_notional = price * 1000.0
        if contract_notional <= 0:
            return

        target_contracts = int(round(target_notional / contract_notional))
        current_contracts = algo.Portfolio[symbol].Quantity
        delta = target_contracts - current_contracts

        if delta != 0:
            algo.MarketOrder(symbol, delta)
# Agent 2: ES Dynamic Hedge Agent (RL Actor–Critic)

class ESHedgeAgent:
    """
    RL environment for dynamic ES hedging against VX exposure.

    State (4 dims):
        [ vx_exposure,
          vvix_level,
          vvix_change,
          vix_change ]

    Action space:
        Discrete hedge ratios mapped via HEDGE_RATIOS:
            0 -> 0.0   (no hedge)
            1 -> 0.3   (30% notional)
            2 -> 0.7   (70% notional)
            3 -> 1.0   (100% notional)

    Reward (for learning):
        - Same equity delta as VIX agent (shared portfolio),
        - plus actor–critic baseline.
    """

    HEDGE_RATIOS = [0.0, 0.3, 0.7, 1.0]

    def __init__(self, algorithm: QCAlgorithm, trader_agent: VIXTraderAgent):
        self.algorithm = algorithm
        self.trader = trader_agent  # reference to VIX RL agent

        # Allocation from risk manager (0..1)
        self.allocation = 0.0

        # Original state components
        self.vvix = None
        self.last_vvix = None
        self.vvix_change = 0.0

        self.vix = None
        self.last_vix = None
        self.vix_change = 0.0

        self.front_es_symbol = None
        self.prev_es_symbol = None

        self.current_obs = None
        self.has_state = False

        # RL-specific
        self.obs_dim = 4
        self.num_actions = len(self.HEDGE_RATIOS)
        self.device = torch.device("cpu")

        self.net = ActorCriticNet(self.obs_dim, self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        self.last_obs_tensor = None
        self.last_action_tensor = None
        self.last_equity = None

        self.reward_scale = 10000.0
        self.entropy_coef = 0.001
        self.critic_coef = 0.5

    def SaveModel(self, key: str):
        import pickle
        data = pickle.dumps(self.net.state_dict())
        self.algorithm.ObjectStore.SaveBytes(key, data)
        self.algorithm.Debug(f"Saved ES model to ObjectStore as {key}")

    def LoadModel(self, key: str):
        import pickle
        data = self.algorithm.ObjectStore.ReadBytes(key)
        state_dict = pickle.loads(data)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.algorithm.Debug(f"Loaded model from ObjectStore: {key}")

    # ----- External hooks -----
    def SetAllocation(self, allocation: float):
        """Called by risk manager to scale hedge notional."""
        self.allocation = max(0.0, min(1.0, allocation))

    # ----- Internal index + contract updates -----
    def _update_indices(self, data: Slice):
        algo = self.algorithm

        # Use cached daily values when available
        if algo.last_vvix is not None:
            self.vvix = algo.last_vvix
        if algo.last_spot_vix is not None:
            self.vix = algo.last_spot_vix

        # Update deltas only if new daily bar present in Slice
        if data.ContainsKey(algo.vvix_index):
            self.vvix = float(data[algo.vvix_index].Close)
            if self.last_vvix is None:
                self.last_vvix = self.vvix
            self.vvix_change = self.vvix - self.last_vvix
            self.last_vvix = self.vvix

        if data.ContainsKey(algo.vix_index):
            self.vix = float(data[algo.vix_index].Close)
            if self.last_vix is None:
                self.last_vix = self.vix
            self.vix_change = self.vix - self.last_vix
            self.last_vix = self.vix

    def _update_front_es_contract(self, data: Slice):
        algo = self.algorithm
        chains = data.FutureChains
        if algo.es_future_symbol not in chains:
            return

        chain = chains[algo.es_future_symbol]
        contracts = sorted(chain, key=lambda x: x.Expiry)
        if not contracts:
            return

        self.front_es_symbol = contracts[0].Symbol

    # ----- State collection -----
    def CollectState(self, data: Slice):
        """
        Called from MultiAgentSystem.OnData every bar to keep obs in sync.
        """
        self._update_indices(data)
        self._update_front_es_contract(data)

        if self.vvix is None or self.vix is None:
            self.has_state = False
            self.current_obs = None
            return

        algo = self.algorithm
        vx_exposure = 0.0
        if getattr(self.trader, "front_contract_symbol", None) is not None:
            vx_exposure = float(algo.Portfolio[self.trader.front_contract_symbol].Quantity)

        self.current_obs = np.array([
            vx_exposure,
            float(self.vvix),
            float(self.vvix_change),
            float(self.vix_change)
        ], dtype=np.float32)

        self.has_state = True

    def RefreshAfterTrader(self):
        """
        Called after VX agent trades, to update vx_exposure component of obs.
        """
        algo = self.algorithm
        if self.current_obs is None:
            return

        vx_exposure = 0.0
        if getattr(self.trader, "front_contract_symbol", None) is not None:
            vx_exposure = float(algo.Portfolio[self.trader.front_contract_symbol].Quantity)

        self.current_obs[0] = vx_exposure
    def UpdateFromReward(self, current_equity: float, train_phase: bool):
        """
        Same pattern as VIX trader: one-step actor–critic update using
        equity delta as reward.
        """
        if (not train_phase or
            self.last_obs_tensor is None or
            self.last_action_tensor is None or
            self.last_equity is None):
            self.last_equity = current_equity
            return

        reward = (current_equity - self.last_equity) / self.reward_scale
        reward_t = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # Fresh forward pass to build a new graph
        logits, value = self.net(self.last_obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(self.last_action_tensor)

        advantage = reward_t - value.detach()
        actor_loss = -logprob * advantage
        critic_loss = F.mse_loss(value, reward_t)
        entropy = dist.entropy().mean()

        loss = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        self.last_equity = current_equity

    # ----- Decide and act -----
    def DecideAndAct(self, train_phase: bool):
        algo = self.algorithm

        if not self.has_state or self.current_obs is None:
            return

        obs = self.current_obs
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits, value = self.net(obs_t)
        dist = torch.distributions.Categorical(logits=logits)

        if train_phase:
            action_tensor = dist.sample()
        else:
            action_tensor = torch.argmax(logits, dim=-1)

        action_index = int(action_tensor.item())

        self.last_obs_tensor = obs_t
        self.last_action_tensor = action_tensor
        self.last_equity = algo.Portfolio.TotalPortfolioValue

        self._execute_action(action_index)

    # ----- Trade execution -----
    def _execute_action(self, action_index: int):
        """
        Translate action index -> hedge ratio -> ES futures position.
        """
        algo = self.algorithm
        ratio = self.HEDGE_RATIOS[action_index]

        if self.front_es_symbol is None or getattr(self.trader, "front_contract_symbol", None) is None:
            return

        # If ES front rolled, liquidate old ES positions
        if self.prev_es_symbol is not None and self.prev_es_symbol != self.front_es_symbol:
            old_es_qty = algo.Portfolio[self.prev_es_symbol].Quantity
            if old_es_qty != 0:
                algo.MarketOrder(self.prev_es_symbol, -old_es_qty)
        self.prev_es_symbol = self.front_es_symbol

        vx_symbol = self.trader.front_contract_symbol
        es_symbol = self.front_es_symbol

        vx_qty = float(algo.Portfolio[vx_symbol].Quantity)

        # If no VX exposure, close any ES hedge
        if vx_qty == 0:
            current_es = algo.Portfolio[es_symbol].Quantity
            if current_es != 0:
                algo.MarketOrder(es_symbol, -current_es)
            return

        vx_price = algo.Securities[vx_symbol].Price
        es_price = algo.Securities[es_symbol].Price
        if vx_price <= 0 or es_price <= 0:
            return

        # Notional exposure in VX
        vx_notional_abs = abs(vx_price * 1000.0 * vx_qty)  # VX multiplier ~1000
        es_multiplier = 50.0

        es_target_notional = vx_notional_abs * ratio * self.allocation

        # Hedge sign: short VX -> long ES, long VX -> short ES
        hedge_sign = -np.sign(vx_qty)

        es_target_contracts = int(round(
            hedge_sign * es_target_notional / (es_price * es_multiplier)
        ))

        current_es_contracts = algo.Portfolio[es_symbol].Quantity
        delta = es_target_contracts - current_es_contracts

        if delta != 0:
            algo.MarketOrder(es_symbol, delta)

# Agent 3: Risk Manager (Rule-based meta-agent)
class RiskManagerAgent:
    def __init__(self, algorithm: QCAlgorithm,
                 trader_agent: VIXTraderAgent,
                 hedger_agent: ESHedgeAgent):

        self.algorithm = algorithm
        self.trader = trader_agent
        self.hedger = hedger_agent

        self.returns_window = deque(maxlen=60)  # ~3 months of daily returns
        self.max_equity = None
        self.drawdown = 0.0

        self.current_obs = None
        self.has_state = False

    def OnNewReturn(self, ret: float):
        algo = self.algorithm
        self.returns_window.append(ret)

        equity = algo.Portfolio.TotalPortfolioValue
        if self.max_equity is None:
            self.max_equity = equity
        else:
            self.max_equity = max(self.max_equity, equity)

        if self.max_equity and self.max_equity > 0:
            self.drawdown = (self.max_equity - equity) / self.max_equity
        else:
            self.drawdown = 0.0

    def CollectState(self, data: Slice):
        if len(self.returns_window) < 10:
            self.has_state = False
            self.current_obs = None
            return

        arr = np.array(self.returns_window)
        sharpe = float(np.mean(arr) / (np.std(arr) + 1e-6))

        vvix_level = self.algorithm.last_vvix if self.algorithm.last_vvix is not None else 85.0
        rv_plus = getattr(self.trader, "last_rv_plus", 0.0)

        self.current_obs = np.array([
            sharpe,
            self.drawdown,
            vvix_level,
            rv_plus
        ], dtype=np.float32)

        self.has_state = True

    def DecideAction(self) -> int:
        """
        Simple rule-based allocator:
          0: full VIX, no ES
          1: balanced (0.7 VIX, 0.3 ES)
          2: cautious (0.4 VIX, 0.6 ES)
          3: defensive (0.0 VIX, 1.0 ES)
        """
        if not self.has_state or self.current_obs is None:
            return 1

        sharpe, dd, vvix_level, rv_plus = self.current_obs

        low_dd = 0.05
        med_dd = 0.12
        low_vvix = 80
        med_vvix = 95
        high_vvix = 110

        if dd < low_dd and vvix_level < low_vvix and rv_plus < 0.003 and sharpe > 0:
            return 0

        if dd < med_dd and vvix_level < med_vvix and rv_plus < 0.01:
            return 1

        if dd < 0.25 or vvix_level < high_vvix:
            return 2

        return 3

    def DecodeAllocationAction(self, action_index: int):
        mapping = {
            0: (1.0, 0.0),
            1: (0.7, 0.3),
            2: (0.4, 0.6),
            3: (0.0, 1.0)
        }
        return mapping.get(int(action_index), (0.7, 0.3))