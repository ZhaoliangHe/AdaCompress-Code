import matplotlib.pyplot as plt
from PIL import Image
import pickle as pk
import numpy as np
from collections import defaultdict
import time
from res_manager import ResultManager

def plot_durations(y, title_list):
    plot_count = len(title_list)
    for idx, title in enumerate(title_list):
        plt.subplot(plot_count, 1, idx + 1)
        plt.plot(y[idx, :],'.')
        plt.ylabel(title)
    plt.show()

def evaluation_framework(dnim_on_imagenet):
    smooth_window = 30
    dnim_on_imagenet = dnim_on_imagenet
    fine_grain_accs = dnim_on_imagenet['accuracy']
    fine_grain_probs = dnim_on_imagenet['explor_rate']
    fine_grain_sizes = dnim_on_imagenet['upload_size']#[:1670] + dnim_on_imagenet['comp_size'][1670:]
    fine_grain_refsizes = dnim_on_imagenet['ref_size']
    def list_split(l, size):
        return [l[m:m + size] for m in range(0, len(l), size)]
    aver_accs = [np.mean(item) for item in list_split(fine_grain_accs, smooth_window)]
    aver_probs = [np.mean(item) for item in list_split(fine_grain_probs, smooth_window)]
    aver_sizes = [np.mean(item) for item in  list_split(fine_grain_sizes, smooth_window)]
    aver_ref_sizes = [np.mean(item) for item in  list_split(fine_grain_refsizes, smooth_window)]

    norm_sizes = np.array(aver_sizes) / 200000
    norm_refsizes = np.array(aver_ref_sizes) / 150000

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # r1 = ax.fill_between(x=[i for i in np.arange(0,26,1)], y1=norm_refsizes[:26], y2=1.8, color='oldlace', label="inference")
    # r2 = ax.fill_between(x=[i for i in np.arange(25,56,1)], y1=retrain_y1, y2=1.8, color='oldlace', label="inference")
    # r3 = ax.fill_between(x=[i for i in np.arange(55,114,1)], y1=norm_refsizes[55:], y2=1.8, color='oldlace', label="inference")

    # r1 = ax.fill_betweenx(x1=0, x2=740 / smooth_window, y=[-0.4,1.8], color='oldlace', label="inference")
    # r2 = ax.fill_betweenx(x1=741 / smooth_window, x2=1670 / smooth_window, y=[-0.4, 1.8], color='powderblue', label="retrain")
    # r3 = ax.fill_betweenx(x1=1671 / smooth_window, x2=3097 / smooth_window, y=[-0.4, 1.8], color='oldlace', label="inference")

    # r1 = ax.fill_betweenx(x1=0, x2=926 / smooth_window, y=[-0.4,1.8], color='oldlace', label="inference")
    # r2 = ax.fill_betweenx(x1=927 / smooth_window, x2=1879 / smooth_window, y=[-0.4, 1.8], color='powderblue', label="retrain")
    # r3 = ax.fill_betweenx(x1=1880 / smooth_window, x2=3440 / smooth_window, y=[-0.4, 1.8], color='oldlace', label="inference")

    r1 = ax.fill_betweenx(x1=0, x2=890 / smooth_window, y=[-0.4,1.8], color='oldlace', label="inference")
    r2 = ax.fill_betweenx(x1=891 / smooth_window, x2=1860 / smooth_window, y=[-0.4, 1.8], color='powderblue', label="retrain")
    r3 = ax.fill_betweenx(x1=1861 / smooth_window, x2=3440 / smooth_window, y=[-0.4, 1.8], color='oldlace', label="inference")

    l1 = ax.plot(aver_accs, label=r"recent average accuracy $\bar{\mathcal{A}_t}$", color="green")

    num1 = len(aver_accs)
    rs1 = ax.fill_between(x=[i for i in range(num1)], y1=0, y2=norm_refsizes, alpha=0.5, label="ref_size")
    rs2 = ax.fill_between(x=[i for i in range(num1)], y1=0, y2=norm_sizes, alpha=0.5, label="upload_size")

    ax.set_ylabel(r"Average accuracy $\bar{\mathcal{A}}$")
    ax.set_ylim(0, 1.1)
    ax.set_yticks(ax.get_yticks()[3:-1])

    ax2 = ax.twinx()
    l2 = ax2.plot(aver_probs, label=r"estimation probability $p_{\rm est}$", color="gray", linewidth=0.5)
    ax2.set_ylabel(r"Estimate prob. $p_{\rm est}$")
    ax2.set_ylim(-0.7, 1.1)
    ax2.set_yticks(ax2.get_yticks()[4:-1])

    # ax2.vlines(x=24, ymin=-0.7, ymax=1.8, linestyles='-', linewidth=0.5, color='r')
    ax2.vlines(x=29, ymin=-0.7, ymax=1.8, linestyles='-', linewidth=0.5, color='r')

    plt.xlim(0, 113)
    ax2.set_xlabel("steps")

    # ax2.set_xticks([0, 20, 24, 40, 60, 80, 100, 110])
    # ax2.set_xticklabels(["0", "600", r"$\Delta$", "1200", "1800", "2400", "3000","3300"])
    ax2.set_xticks([0, 20, 29, 40, 60, 80, 100])
    ax2.set_xticklabels(["0", "600", r"$\Delta$", "1200", "1800", "2400", "3000"])

    plt.setp(ax2.get_xticklabels(), rotation=90)

    fig.legend([r1, r2, rs1, rs2] + l1 + l2,
               ["Inference", "Retrain", "Benchmark upload size", "Our upload size", "Accuracy", "Estimate probability"],
               loc=8, ncol=3)

    plt.subplots_adjust(top=1.0,
                        bottom=0.33,
                        left=0.105,
                        right=0.91,
                        hspace=0.2,
                        wspace=0.2)
    plt.show()

x = np.linspace(0, 2 * np.pi, 50)
y = np.sin(x)
plt.plot(x, y)
plt.show()

rm = ResultManager('evaluation_results')
rm.print_meta_info()
not_reload = rm.load(16)
reload = rm.load(20)

initial_log = reload
# plot_keys = ['status', 'step_count', 'comp_size', 'upload_size']
# plot_keys = ['agent_epsilon', 'agent_accuracy', 'recent_reward', 'explor_rate']
plot_keys = ['agent_epsilon', 'recent_reward', 'explor_rate']
plot_durations(np.array([initial_log[key] for key in plot_keys]),
               title_list=plot_keys)
plot_keys = ['status', 'recent_accuracy', 'accuracy', 'upload_size']
plot_durations(np.array([initial_log[key] for key in plot_keys]),
               title_list=plot_keys)
# # evaluation_framework(not_reload)
evaluation_framework(rm.load(17))
recent_accuracy = reload['recent_accuracy']
accuracy = reload['accuracy']

m = 10
accuracy[:m]