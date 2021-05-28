import datetime
import logging
import os
import pprint

from torch.utils.tensorboard import SummaryWriter

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

DATE_FORMAT = "%m-%d-%Y %H:%M:%S"


# TODO: LOG 添加多个实用方法
class Log:
    def __init__(self):
        self.logger_get_count = 0

    def setup(self, log_config):
        # self.time_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_now = log_config.time_now

        self.tensorboard_dir = os.path.join(log_config.tb_dir, self.time_now)
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)

        self.log_dir = os.path.join(log_config.log_dir, self.time_now)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.writer = SummaryWriter(self.tensorboard_dir, flush_secs=2)

        self.logger = self.createlog(log_config.log_name)

    def config_info(self, config_list):
        with open(self.fname, "a") as f:
            for config_dict in config_list:
                pprint.pprint("=======  " + config_dict.config_name + "  ========", f)
                # pprint.pprint('\n=====================\n',f)
                pprint.pprint(config_dict.get_config_dict(), f)

    def add_scalar(self, name, value_y, value_x):
        """使用tensorboard记录学习过程

        Args:
            name (string): 学习曲线的名字，一般格式为："name/subname"
            value_y (scalar): y value
            value_x (scalar): x value
        """
        self.writer.add_scalar(name, value_y, value_x)
        self.writer.flush()

    def add_dict(self, log_dict):
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v[0], v[1])
        self.writer.flush()

    def info0(self, info_list):
        """把info_list中的信息写入日志文件

        Args:
            info_list (lsit): 日志行信息，格式为：[a, b, c], a,b,c 可为任意类型
        """
        string_info = " ".join([str(i) for i in info_list])
        self.logger.info(string_info)

    def info(self, *args, stdout=False):
        # TODO: add color params
        log_info = [str(i) for i in args]
        string_info = " ".join(log_info)
        self.logger.info(string_info)
        if stdout:
            print(string_info)

    # def league_info(self, info_list):
    #     """把info_list中的信息写入日志文件

    #     Args:
    #         info_list (lsit): 日志行信息，格式为：[a, b, c], a,b,c 可为任意类型
    #     """
    #     string_info = " ".join([str(i) for i in info_list])
    #     self.logger.info(string_info)

    def createlog(self, name):
        # import coloredlogs
        logger = logging.getLogger(name)

        FIELD_STYLES = dict(
            asctime=dict(color="green"),
            hostname=dict(color="magenta"),
            levelname=dict(
                color="green",
            ),
            filename=dict(color="magenta"),
            name=dict(color="blue"),
            threadName=dict(color="green"),
        )

        LEVEL_STYLES = dict(
            debug=dict(color="yellow"),
            info=dict(color="blue"),
            warning=dict(color="yellow"),
            error=dict(color="red"),
            critical=dict(
                color="red",
            ),
        )

        logger.setLevel(logging.DEBUG)
        # coloredlogs.install(
        #     level="DEBUG",
        #     logger=logger,
        #     # fmt="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        #     fmt="%(asctime)s - %(levelname)s  %(message)s",
        #     level_styles=LEVEL_STYLES,
        #     field_styles=FIELD_STYLES)

        self.fname = self.log_dir + "/" + name + "_" + self.time_now + ".log"

        file_handler = logging.FileHandler(self.fname)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        # stdout_handler = logging.StreamHandler(sys.stdout)
        # stdout_handler.setFormatter(logging.Formatter(LOG_FORMAT,DATE_FORMAT))
        logger.handlers.append(file_handler)
        # logger.handlers.append(stdout_handler)
        return logger


logger = Log()


def Logger(config=None):
    if logger.logger_get_count == 0:
        assert config is not None, "第一次使用logger，必须用config初始化Log"
        logger.setup(config)
        logger.logger_get_count = 1
    return logger
