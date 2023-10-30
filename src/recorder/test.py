def test_csv_log() -> None:
    from src.recorder.v1 import Recorder

    recorder = Recorder("")
    recorder.csv_log({"test": 2})


test_csv_log()
