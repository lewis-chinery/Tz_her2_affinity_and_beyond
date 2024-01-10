import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import pylab as pl
from IPython import display


def plot_enrichment(enrichments_latest_round, enrichments_cumulative, max_num_rounds,
                    num_sequences_to_sample):
    '''
    
    '''
    rounds = [_ for _ in range(1, max_num_rounds+1)]

    # initiate plot with labels
    plt.clf()
    mpl.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(4, 4), dpi=100)
    plt.xlabel(f"Enrichment round\n(1 round = {num_sequences_to_sample} seqs)")
    plt.ylabel("Binder enrichment (%)")
    plt.ylim(0, 105)
    plt.xlim(0.5, max_num_rounds+0.5)

    # plot enrichments for all currently processed rounds
    plt.plot(rounds, enrichments_latest_round, color='green', marker='.', linestyle='-', linewidth=2, markersize=10, label="Most recent round")
    plt.plot(rounds, enrichments_cumulative, color='grey', marker='.', linestyle='-', linewidth=2, markersize=10, label="Cumulative")
    plt.legend(loc="upper left", fontsize=14)
    display.clear_output(wait=True)
    display.display(pl.gcf())
    time.sleep(1.0)
