#!/usr/bin/env python3

import algorithms_cumulative

import argparse
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time
import matplotlib.colors as mcolors
import tqdm
import re
import seaborn as sns

from multiprocessing import Pool


# OPT must be the first algorithm in this list
# OPT_multiple must be the second
ALGORITHMS_ONLINE = (
  algorithms_cumulative.OPT,
  # algorithms.OPT_multiple,
  # algorithms.Online_multiple,
  # algorithms.RandomOnline_multiple,
  # algorithms.RandomOnline_multiple_prudent,
  algorithms_cumulative.ClassicDet,
  algorithms_cumulative.ClassicRandom,
  # algorithms.Rent,
)

ALGORITHMS_PRED= (
  algorithms_cumulative.FTP,
  algorithms_cumulative.OLDynamic,
  algorithms_cumulative.OLStatic,
  algorithms_cumulative.Buy,
  algorithms_cumulative.PACPredictRand1,
  # algorithms_cumulative.KumarRandom,
  # algorithms.FTP_multiple,
  # algorithms.RhoMu_paretomu_1_05,
  # algorithms.RhoMu_paretomu_1_1,
  # algorithms.RhoMu_paretomu_1_1596,
  # algorithms.RhoMu_paretomu_1_216,
  # algorithms.RhoMu_paretomu_1_3,
  # algorithms.RhoMu_paretomu_1_4,
  # algorithms.RhoMu_paretomu_1_5,
  # algorithms.RhoMu_multiple_1_1596,
  # algorithms.RhoMu_multiple_1_216,
  # algorithms_cumulative.RobustRhoMu,
  # algorithms.RobustRhoMu_multiple,
  # algorithms.RobustRhoMu_multiple_prudent,
  # algorithms.KumarRandom_1_1596,
  # algorithms.KumarRandom_1_216,
  # algorithms.Kumar_multiple_1_1596,
  # algorithms.Kumar_multiple_1_216,
  # algorithms.RobustKumar,
  # algorithms.RobustKumar_multiple,
  # algorithms.RobustKumar_multiple_prudent,
  # algorithms.AngelopoulosDet_1_1596,
  # algorithms.AngelopoulosDet_1_216,
  # algorithms.RobustAngelo,
  # algorithms.RobustAngelo_multiple,
  # algorithms.RobustFTP,
  # algorithms.RobustFTP_multiple_prudent,
)

ALGORITHMS_PAC= (
  # algorithms_cumulative.PACPredict1,
  # algorithms_cumulative.PACPredict2,
  # algorithms_cumulative.PACPredict3,
  # algorithms_cumulative.PACPredictRand1,
  # algorithms_cumulative.PACPredictRand2,
  # algorithms.OnlineLearning,
)

ALGORITHMS_PPP= (
  # algorithms_cumulative.PPPAlgorithm,
)

def RunnerForPool(algorithm, requests, predictions, sigmas):
  return algorithms_cumulative.Cost_history(algorithm(requests, predictions, sigmas), requests, algorithm.__name__)
  # return algorithms_cumulative.CR_history(algorithm(requests, predictions, sigmas), requests, algorithm.__name__)

def PlotPredRandom(datasets, preds, num_runs, output_basename=None, load_json=None, keep='rhos', res_type="cr"):
  d = 0.1
  ALGORITHMS = ALGORITHMS_ONLINE + ALGORITHMS_PRED
  with open(datasets[0]) as f:
    MAX_REQUEST = np.ceil(max(float(line) for line in f))
  print(MAX_REQUEST)
  SIGMA = MAX_REQUEST
  
  for dataset in datasets:
      with open(dataset) as f:
        requests = tuple(f)
        requests = requests[:3000]
      assert(len(requests) > 0)
      
      requests = [float(x) for x in requests]

  costs_history = np.zeros((len(ALGORITHMS), len(requests), num_runs))
  # predictor = lambda sigma: algorithms.PredRandom(yhat, sigma)
  predictor = lambda sigma: algorithms_cumulative.PredRandom(requests, sigma)

  sig = 6
  # # PREDICTIONS
  # # Generated sequence of random predictions (based on variance sequence)
  predictions = [predictor(([0]*10+[sig]*10) * int(len(requests) / 20)) for _ in range(num_runs)]

  for i, algorithm in enumerate(ALGORITHMS_ONLINE):
      output = algorithm(requests)
      cost = algorithms_cumulative.Cost_history(output,requests, algorithm.__name__)
      # cost = algorithms_cumulative.CR_history(output,requests, algorithm.__name__)
      # CUMULATIVE LOSS
      for run in range(num_runs):
        costs_history[i,:,run] = cost

  # # SIGMAS
  # Construct list of sigmas for every timestep
  # This should match the sequences of sigmas inputted into PREDICTIONS above
  sigmas = ([0]*10+[sig]*10) * int(len(requests) / 20)

  with Pool() as pool:
    grid = list(itertools.product(range(len(ALGORITHMS_PRED)), range(num_runs)))
    args = [(ALGORITHMS_PRED[algorithm_idx], requests, predictions[run_idx], sigmas)
            for algorithm_idx, run_idx in grid]
    runs = pool.starmap(RunnerForPool, args)
    for (algorithm_idx, run_idx), cost in zip(grid, runs):
      costs_history[len(ALGORITHMS_ONLINE) + algorithm_idx,:,run_idx] = cost
  print(costs_history.shape)
  cumulative_losses = costs_history - costs_history[0]
  # cumulative_losses = costs_history
  print('Maximum std dev among runs: %0.5f' % np.max(np.std(cumulative_losses, axis=2)))
  cumulative_losses = np.mean(cumulative_losses, axis=2)

  LABELS = [algorithm.__name__ for algorithm in ALGORITHMS]
  if (keep=='rhos'):
    OPT_LABEL = 0
    KEEP = {
      "ClassicRandom" : "$(\\frac{e}{e-1})$-competitive", # "Classical (randomized)",
      "KumarRandom_1_1596" : "PSK ($\\rho$=1.1596)",
      "KumarRandom_1_216" : "PSK ($\\rho$=1.216)",
      "RhoMu_paretomu_1_1596" : "Our alg. ($\\rho$=1.1596)",
      "RhoMu_paretomu_1_216" : "Our alg. ($\\rho$=1.216)",
      "AngelopoulosDet_1_1596" : "ADJKR ($\\rho$=1.1596)",
      "AngelopoulosDet_1_216" : "ADJKR ($\\rho$=1.216)",
      "FTP" : "FTP",  
    }
  elif (keep=='ouralg_manyrhos'):
    OPT_LABEL = 0
    KEEP = {
      "ClassicRandom" : "$(\\frac{e}{e-1})$-competitive", # "Classical (randomized)",
      "RhoMu_paretomu_1_1" : "Our alg. ($\\rho$=1.1)",
      "RhoMu_paretomu_1_1596" : "Our alg. ($\\rho$=1.1596)",
      "RhoMu_paretomu_1_3" : "Our alg. ($\\rho$=1.3)",
      "RhoMu_paretomu_1_4" : "Our alg. ($\\rho$=1.4)",
      "RhoMu_paretomu_1_5" : "Our alg. ($\\rho$=1.5)",
      "FTP" : "FTP",   
    }
  elif (keep=='best'):
    OPT_LABEL = 0
    KEEP = {
      "ClassicRandom" : "WOA",
      # "ClassicRandom" : "Classical (randomized)",
      # "ClassicDet" : "Classical (deterministic)", 
      "FTP" : "FTP",  
      # "RobustRhoMu" : "Antoniadis algorithm",
      "OLDynamic": "OL-Dynamic",
      "OLStatic": "OL-Static",
      # "Buy": "Buy",
      # "PACPredict1": "PAC algorithm (0.1)",
      # "PACPredict2": "PAC algorithm (0.05)",
      # "PACPredict3": "PAC algorithm (0.01)",
      "PACPredictRand1": "RSR-PIP",
      # "PACPredictRand2": "Rand PAC algorithm (0.01)",
      # "PPPAlgorithm": "PPP algorithm (0.5)",
    }
  elif (keep=='prudent'):
    OPT_LABEL = 1
    KEEP = {
      "RandomOnline_multiple_prudent" : "$(\\frac{e}{e-1})$-competitive (prudent)",
      "RandomOnline_multiple" : "$(\\frac{e}{e-1})$-competitive (not prudent)",
      "RobustKumar_multiple_prudent": "Lemma 3 + Thm 28 + PSK (prudent)",
      "RobustKumar_multiple": "Lemma 3 + Thm 28 + PSK (not prudent)",
      "RobustAngelo_multiple": "Lemma 3 + Thm 28 + ADJKR",
      "RobustRhoMu_multiple_prudent": "Our algorithm (prudent)",
      "RobustRhoMu_multiple": "Our algorithm (not prudent)",
    }
  else:  # keep == 'bestmult'
    OPT_LABEL = 1
    KEEP = { 
      "RandomOnline_multiple_prudent" : "$(\\frac{e}{e-1})$-competitive", # "Classical (randomized)",
      "FTP_multiple": "Lemma 3 + FTP",
      "RobustFTP_multiple_prudent": "Lemma 3 + Thm 28 + FTP",
      "RobustKumar_multiple_prudent": "Lemma 3 + Thm 28 + PSK",
      "RobustAngelo_multiple": "Lemma 3 + Thm 28 + ADJKR",
      "RobustRhoMu_multiple_prudent": "Our algorithm",
    }
  

  # GENERAL_FONTSIZE, LABEL_FONTSIZE, LEGEND_FONTSIZE = 9, 12, 10
  GENERAL_FONTSIZE, LABEL_FONTSIZE, LEGEND_FONTSIZE = 11, 16, 11
  
  plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': GENERAL_FONTSIZE,
    'pgf.rcfonts': False,
  })
  LINE = iter(itertools.cycle(itertools.product(['', '.', '+', 'x'], ['--', '-.', '-', ':'])))

  if keep == 'ouralg_manyrhos':
    LINE = iter(itertools.cycle((('','--'),)))
  elif keep == 'best':
    LINE = iter([('',':'), ('','-.'), ('','--'), ('', '--'), ('', '-'), ('', '-'), ('', '-'),
                  ('x', (1, (1, 2))), ('.', '--'), ('.', (2, (1, 2)))])
  
  COLORS = iter(mcolors.TABLEAU_COLORS)
  
  oldlabel = ""
  oldcolor = ""
  start_mk, start_ls = "X","X"
  labels = []
  data = []
  for label, losses in zip(LABELS, cumulative_losses):
    if label not in KEEP:
      continue
    label = KEEP[label]
    labels.append(label)
    data.append(list(losses))
    if (label.split('[')[0] == oldlabel.split('[')[0]):
      color = oldcolor
      mk, ls = next(LINE)
    elif "rho=" in label and start_mk != "X":
      while (start_ls != ls):
        mk,ls = next(LINE)
    elif "rho=" in label and start_mk == "X":
      mk, ls = next(LINE)
      start_mk = mk
      start_ls = ls
    else:
      print(label)
      mk, ls = next(LINE)

    if (label.split('[')[0] != oldlabel.split('[')[0]):
      color = next(COLORS)
    
    oldlabel = label
    oldcolor = color
    data = list(losses)
    plt.plot(range(len(requests)), data, label=label, markersize=5, marker=mk, ls=ls, color=color)[0]
  xlabel = plt.xlabel('time (t)', fontsize=LABEL_FONTSIZE)
  ylabel = plt.ylabel('Cumulative Excess CR', fontsize=LABEL_FONTSIZE)
  
  plt.gcf().set_size_inches(w=6, h=4.5)
  plt.tight_layout()
  if keep == 'ouralg_manyrhos':
    pass
  else:
    plt.legend(loc='best', ncol=(1 if len(KEEP) < 8 else 2), fontsize=LEGEND_FONTSIZE)

  if output_basename:
    plt.savefig(output_basename + '.pdf')
  else:
    plt.show()

  # print(np.mean(data, axis=1))
  # print(labels)

  # Remove outliers
  # extr = np.percentile(data, 99, axis=1)
  # for i in range(len(data)):
  #   loss = np.array(data[i])
  #   print(extr[i])
  #   data[i] = loss[loss < extr[i]]
  
  # quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
  # # q1, m, q3 = np.percentile(data[2], [25,50,75])
  # # print(q1, m, q3)
  # print(quartile1, medians, quartile3)

  # inds = np.arange(1, len(medians) + 1)
  # plt.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
  # plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)

  # fig = plt.violinplot(data, showextrema=False)
  # plt.xticks(np.arange(1,len(labels)+1),labels)
  # ylabel = plt.ylabel('CR', fontsize=LABEL_FONTSIZE)
  # plt.ylim(0.5,15)
  # plt.yscale("log")
  # plt.gcf().set_size_inches(w=6, h=4.5)
  # plt.tight_layout()
  # for pc in fig['bodies']:
  #   # pc.set_facecolor('#D43F3A')
  #   # pc.set_edgecolor('black')
  #   pc.set_alpha(0.5)

  # plt.show()

  # if output_basename:
  #   plt.savefig(output_basename + '.pdf')
  # else:
  #   plt.show()

  # print(labels)
  # ftp = data[1]
  # print(ftp[:100])
  # print(cumulative_losses[0][:100])
  
  # sns.stripplot(data=data, size=1)
  # for d in data:
  #   lst = np.array(d)
  #   print(lst[lst<1])
  # sns.violinplot(data=data, cut=0, bw_adjust=0.5)
  # # plt.ylim(0.5,10)
  # plt.xticks(np.arange(0,len(labels)),labels)
  # plt.show()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--res_type', type=str, default='cr', help="competitive ratio or regret")
  parser.add_argument('-n', '--num_runs', type=int, default=1, help='number of runs')
  parser.add_argument('-o', '--output_basename', type=str)
  parser.add_argument('-l', '--load_json', type=str)
  parser.add_argument('-k', '--keep', type=str, choices=['rhos', 'best', 'bestmult', 'ouralg_manyrhos', 'prudent'], default='rhos')
  parser.add_argument('-p', '--predictions', default=None, type=str, nargs='+')
  parser.add_argument('-d', '--datasets', type=str, nargs='+')

  args = parser.parse_args()

  PlotPredRandom(args.datasets, args.predictions, args.num_runs, args.output_basename, args.load_json, args.keep, args.res_type)

if __name__ == '__main__':
  main()
