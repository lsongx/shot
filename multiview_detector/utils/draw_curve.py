import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt


def draw_curve(path, x_epoch, train_loss, train_prec, test_loss, test_prec, test_moda=None):
    fig, axes = plt.subplots(1,3)
    # ax1 = fig.add_subplot(131, title="loss")
    # ax2 = fig.add_subplot(132, title="prec")
    axes[0].plot(x_epoch, train_loss, 'bo-', label='train' + ': {:.3f}'.format(train_loss[-1]))
    axes[0].plot(x_epoch, test_loss, 'ro-', label='test' + ': {:.3f}'.format(test_loss[-1]))
    axes[1].plot(x_epoch, train_prec, 'bo-', label='train' + ': {:.1f}'.format(train_prec[-1]))
    axes[1].plot(x_epoch, test_prec, 'ro-', label='test' + ': {:.1f}'.format(test_prec[-1]))

    axes[0].legend()
    axes[1].legend()
    if test_moda is not None:
        # axes[2] = fig.add_subplot(133, title="moda")
        axes[2].plot(x_epoch, test_moda, 'ro-', label='test' + ': {:.1f}'.format(test_moda[-1]))
        axes[2].legend()
    fig.savefig(path)
