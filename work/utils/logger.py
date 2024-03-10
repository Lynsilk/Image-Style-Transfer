import logging

class Logger(logging.Logger):
    def __init__(self, name='cycleGAN', level=logging.INFO, save_root:str=None):
        super(Logger, self).__init__(name, level)

        formatter = logging.Formatter('%(asctime)s - %(message)s')
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.addHandler(console)

        if save_root is not None:
            handler = logging.FileHandler("{}/log".format(save_root))
            handler.setLevel(logging.INFO)
            self.addHandler(handler)
