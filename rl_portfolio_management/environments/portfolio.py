import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint
import logging
import os
import tempfile
import time
import gym
import gym.spaces
import random
from scipy import stats

from ..config import eps
from ..data.utils import normalize, random_shift, scale_to_start
from ..util import MDD as max_drawdown, sharpe, softmax
from ..callbacks.notebook_plot import LivePlotNotebook

logger = logging.getLogger(__name__)


class DataSrc(object):
    """Acts as data provider for each new episode."""

    def __init__(self, steps=252, scale=True, scale_extra_cols=True, augment=0.00, window_length=50):
        """
        DataSrc.

        df - csv for data frame index of timestamps
             and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close',...]]
             an example is included as an hdf file in this repository
        steps - total steps in episode
        scale - scale the data for each episode
        scale_extra_cols - scale extra columns by global mean and std
        augment - fraction to augment the data by
        """
        self.steps = steps + 1
        self.augment = augment
        self.scale = scale
        self.scale_extra_cols = scale_extra_cols
        self.asset_names = []
        self.features = []
        self.window_length = window_length
        self.gbm = GBM()

        # get rid of NaN's
        self.reset()

    def _step(self):
        # get history matrix from dataframe
        # (eq.1) prices
        data_window = self.data[self.step:self.step+self.window_length+1]

        y1 = data_window[-1] / data_window[-2]

        self.step += 1
        history = data_window[:-1]
        done = bool(self.step >= self.steps)

        return history, y1, done

    def reset(self):
        self.step = 0

        datas = []
        for i in range(self.window_length + self.steps):
            datas.append(self.gbm.get_next_value())
        self.data = random_sequence(self.window_length+self.steps)


class PortfolioSim(object):
    """
    Portfolio management sim.

    Params:
    - cost e.g. 0.0025 is max in Poliniex

    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=[], steps=128, trading_cost=0.0025, time_cost=0.0, mdp_type="MDP"):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names
        self.reset()
        self.mdp_type = mdp_type

    def _step(self, w1, y1):
        """
        Step.

        w1 - new action of portfolio weights
        y1 - price relative vector also called return
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        w0 = self.w0
        p0 = self.p0

        # (eq16) cost to change portfolio
        # (excluding change in cash to avoid double counting for transaction cost)
        c1 = self.cost * (np.abs(w0 - w1))

        p1 = p0 * (1 - c1) * (1-w0 + np.dot(y1, w0))  # (eq11) final portfolio value

        rho1 = p1 / p0 - 1  # rate of returns

        r1 = np.log((p1 + eps) / (p0 + eps))  # (eq10) log rate of return
        if self.mdp_type == "MMDP":
            reward = r1
        else:
            reward = p1 - p0

        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done
        done = bool(p1 <= 0)

        # should only return single values, not list
        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": c1,
        }
        # record weights and prices
        info['weight'] = w1
        info['price'] = y1

        self.infos.append(info)
        return reward, info, done, p0

    def reset(self):
        self.infos = []
        self.w0 = np.array([0])
        self.p0 = np.array([1])


class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.

    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.

    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['notebook', 'ansi']}

    def __init__(self,
                 df,
                 steps=256,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 augment=0.00,
                 output_mode='EIIE',
                 log_dir=None,
                 scale=True,
                 scale_extra_cols=True,
                 ):
        """
        An environment for financial portfolio management.

        Params:
            df - csv for data frame index of timestamps
                 and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close']]
            steps - steps in episode
            window_length - how many past observations["history"] to return
            trading_cost - cost of trade as a fraction,  e.g. 0.0025 corresponding to max rate of 0.25% at Poloniex (2017)
            time_cost - cost of holding as a fraction
            augment - fraction to randomly shift data by
            output_mode: decides observation["history"] shape
            - 'EIIE' for (assets, window, 3)
            - 'atari' for (window, window, 3) (assets is padded)
            - 'mlp' for (assets*window*3)
            log_dir: directory to save plots to
            scale - scales price data by last opening price on each episode (except return)
            scale_extra_cols - scales non price data using mean and std for whole dataset
        """
        self.src = DataSrc(steps=steps, scale=scale, scale_extra_cols=scale_extra_cols,
                           augment=augment, window_length=window_length)
        self._plot = self._plot2 = self._plot3 = None
        self.output_mode = output_mode
        self.sim = PortfolioSim(
            asset_names=self.src.asset_names,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)
        self.log_dir = log_dir

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        nb_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.Discrete(2)

        # get the history space from the data min and max
        if output_mode == 'EIIE':
            obs_shape = (
                nb_assets,
                window_length,
                len(self.src.features)
            )
        elif output_mode == 'atari':
            obs_shape = (
                window_length,
                window_length,
                len(self.src.features)
            )
        elif output_mode == 'mlp':
            obs_shape = window_length
        else:
            raise Exception('Invalid value for output_mode: %s' %
                            self.output_mode)

        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(
                -10,
                20 if scale else 1,  # if scale=True observed price changes return could be large fractions
                obs_shape
            ),
            'weights': self.action_space
        })
        self._reset()

    def _step(self, action):
        """
        Step the env.

        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight between 0 and 1. The first (w0) is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        logger.debug('action: %s', action)

        weights = np.clip(action, 0.0, 1.0)
        weights /= weights.sum() + eps

        history, y1, done1 = self.src._step()

        reward, info, done2, p1 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod(
            [inf["market_return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['steps'] = self.src.step

        self.infos.append(info)

        # reshape history according to output mode
        if self.output_mode == 'EIIE':
            pass
        elif self.output_mode == 'atari':
            padding = history.shape[1] - history.shape[0]
            history = np.pad(history, [[0, padding], [
                0, 0], [0, 0]], mode='constant')
        elif self.output_mode == 'mlp':
            history = history.flatten()

        return {'history': history, 'weights': weights, "acc":p1}, reward, done1 or done2, info

    def _reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward, done, info = self.step(action)
        return observation

    def _seed(self, seed):
        np.random.seed(seed)
        return [seed]

    def _render(self, mode='notebook', close=False):
        # if close:
            # return
        if mode == 'ansi':
            pprint(self.infos[-1])
        elif mode == 'notebook':
            self.plot_notebook(close)

    def plot_notebook(self, close=False):
        """Live plot using the jupyter notebook rendering of matplotlib."""

        if close:
            self._plot = self._plot2 = self._plot3 = None
            return

        df_info = pd.DataFrame(self.infos)
        df_info.index = pd.to_datetime(df_info["date"], unit='s')

        # plot prices and performance
        all_assets = ['BTCBTC'] + self.sim.asset_names
        if not self._plot:
            colors = [None] * len(all_assets) + ['black']
            self._plot_dir = os.path.join(
                self.log_dir, 'notebook_plot_prices_' + str(time.time())) if self.log_dir else None
            self._plot = LivePlotNotebook(
                log_dir=self._plot_dir, title='prices & performance', labels=all_assets + ["Portfolio"], ylabel='value', colors=colors)
        x = df_info.index
        y_portfolio = df_info["portfolio_value"]
        y_assets = [df_info['price_' + name].cumprod()
                    for name in all_assets]
        self._plot.update(x, y_assets + [y_portfolio])


        # plot portfolio weights
        if not self._plot2:
            self._plot_dir2 = os.path.join(
                self.log_dir, 'notebook_plot_weights_' + str(time.time())) if self.log_dir else None
            self._plot2 = LivePlotNotebook(
                log_dir=self._plot_dir2, labels=all_assets, title='weights', ylabel='weight')
        ys = [df_info['weight_' + name] for name in all_assets]
        self._plot2.update(x, ys)

        # plot portfolio costs
        if not self._plot3:
            self._plot_dir3 = os.path.join(
                self.log_dir, 'notebook_plot_cost_' + str(time.time())) if self.log_dir else None
            self._plot3 = LivePlotNotebook(
                log_dir=self._plot_dir3, labels=['cost'], title='costs', ylabel='cost')
        ys = [df_info['cost'].cumsum()]
        self._plot3.update(x, ys)

        if close:
            self._plot = self._plot2 = self._plot3 = None


def random_sequence(length, sin_params=[0.1,0.2,0.3,0.6,0.8,2.0], fluctuation_ratio=0.015):
    def generate_function(index_array):
        phase = np.random.randint(0, 50)
        for i in index_array:
            i = int(i)
            index_array[i] = 1 + fluctuation_ratio*(sum([np.sin((i+phase)*param) for param in sin_params]))
        return index_array
    return np.fromfunction(function=generate_function, shape=(length,))



class Frequency(object):
    Second  = 1
    Minute  = 60
    Hour    = 60*60
    Day     = 60*60*24



class GBM(object):
    DAYSINYEAR = 252
    HOURSINYEAR = DAYSINYEAR * 8
    MINUTESINYEAR = HOURSINYEAR * 60
    SECONDSINYEAR = MINUTESINYEAR * 60

    # @ interval is in days
    # @ 1 min frequency is 1/60*24
    def __init__(self, init_value=1, interval=1.0, freq=Frequency.Day):
        self.__frequency = freq

        if (self.__frequency == Frequency.Day):
            self.__interval = float((interval * 1.0) / GBM.DAYSINYEAR)
        elif (self.__frequency == Frequency.Hour):
            self.__interval = float((interval * 1.0) / GBM.HOURSINYEAR)
        elif (self.__frequency == Frequency.Minute):
            self.__interval = float((interval * 1.0) / GBM.MINUTESINYEAR)
        elif (self.__frequency == Frequency.Second):
            self.__interval = float((interval * 1.0) / GBM.SECONDSINYEAR)

        self.__current = init_value
            #         print ("interval %f" % self.__interval)

    def __getRandom(self):
        return random.random()

    def get_next_value(self, mean=0.1, std=0.2):
        # print ("GBM.get_next_value m=%.3f s=%.3f c=%.3f" % (mean,std,currValue))
        # Geometric Brownian Motion
        #       return =  drift + shock
        #         dS_t =  u d_t  +  sigma dW_t
        #  S_t - S_t-1 = u d_t  +  sigma dW_t
        #          S_t = S_t-1 + u d_t  +  sigma dW_t
        next_value = 0.0
        drift = mean * self.__interval
        shock = std * stats.norm.ppf(self.__getRandom()) * np.sqrt(self.__interval)
        next_value = self.__current + drift + shock
        self.__current = next_value
        return next_value

