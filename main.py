import getopt
import sys

from colorama import Fore

from models.gsgan.Gsgan import Gsgan
from models.leakgan.Leakgan import Leakgan
from models.maligan_basic.Maligan import Maligan
from models.mle.Mle import Mle
from models.rankgan.Rankgan import Rankgan
from models.seqgan.Seqgan import Seqgan
from models.textGan_MMD.Textgan import TextganMmd


def set_gan(gan_name):
    gans = dict()
    gans['seqgan'] = Seqgan
    gans['gsgan'] = Gsgan
    gans['textgan'] = TextganMmd
    gans['leakgan'] = Leakgan
    gans['rankgan'] = Rankgan
    gans['maligan'] = Maligan
    gans['mle'] = Mle
    try:
        Gan = gans[gan_name.lower()]
        gan = Gan()
        gan.vocab_size = 5000
        gan.generate_num = 10000
        return gan
    except KeyError:
        print(Fore.RED + 'Unsupported GAN type: ' + gan_name + Fore.RESET)
        sys.exit(-2)


def set_training(gan):
    try:
        gan_func = gan.train_real
    except AttributeError:
        print(Fore.RED + 'Unsupported training setting: ' + Fore.RESET)
        sys.exit(-3)
    return gan_func


def parse_cmd(argv):
    try:
        opts, args = getopt.getopt(argv, "hg:d:")

        opt_arg = dict(opts)
        if '-h' in opt_arg.keys() or '-d' not in opt_arg.keys():
            print(
                'usage: python main.py -g <gan_type> -d <your_data_location>')
            sys.exit(0)
        if not '-g' in opt_arg.keys():
            print('unspecified GAN type, use MLE training only...')
            gan = set_gan('mle')
        else:
            gan = set_gan(opt_arg['-g'])

        gan_func = set_training(gan)
        gan_func(opt_arg['-d'])
    except getopt.GetoptError:
        print('invalid arguments!')
        print('`python main.py -h`  for help')
        sys.exit(-1)


if __name__ == '__main__':
    gan = None
    parse_cmd(sys.argv[1:])
