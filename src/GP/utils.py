from datetime import datetime


def log(msg: str) -> None:
    """
    Logs given message with datetime in console.

    Paramters:
        msg (str): message to be displayed
    """
    print(datetime.now().strftime(f"[%H:%M:%S] {msg}"))
