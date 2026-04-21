import logging

##### ACTUALISER VALEURS IN LOGGERS OFF UNE FOIS COMPRIS
LOGGERS_OFF= [
        "azure",
        "openai",
        "httpx",
        "httpcore",
        "urllib3",
        "azure.core.pipeline.policies.http_logging_policy",
        "watchfiles.main"
    ]
# Source - https://stackoverflow.com/a/56944256
# Posted by Sergey Pleshakov, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-09, License - CC BY-SA 4.0

import logging

class CustomFormatter(logging.Formatter):

    yellow = "\x1b[38;5;11m"
    orange = "\x1b[38;5;9m"
    purple = "\x1b[38;5;5m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(name)s - %(message)s"

    FORMATS = {
        logging.DEBUG: yellow + format + reset,
        logging.INFO: orange + format + reset,
        logging.WARNING: purple + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(logger_name: str, logs_on: bool = False) -> logging.Logger:
    for name in LOGGERS_OFF:
        target_logger = logging.getLogger(name)
        target_logger.setLevel(logging.ERROR)
    

    logger = logging.getLogger(logger_name)
    if logs_on:
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            #formatter = logging.Formatter('%(name)s - %(message)s')
            formatter = CustomFormatter()
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
    else:
        logging.Logger.disabled = True
    
    return logger

    