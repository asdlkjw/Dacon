from abc import *
from typing import Tuple
from IPython import get_ipython
from IPython.core.ultratb import AutoFormattedTB


class Sender(metaclass=ABCMeta):
    @abstractmethod
    def send(self, msg: str) -> bool:
        raise NotImplementedError


class GlobalExceptionHandler:
    def __init__(
        self,
        except_tuple: Tuple[Exception] = (Exception,),
        sender: Sender = None,
        name: str = "",
    ) -> None:
        """
        예상치 못한 예외를 처리하기 위한 .ipynb 전역 에러 핸들러
        v1은 웹훅을 통해 빠른 알림을 주는게 주 목적

        Args:
            except_tuple: 핸들링할 exception 종류
            sender: 메세지를 보내주는 클래스
            name: 메시지 앞에 붙을 식별자
        """

        self.name = name

        if name != "":
            self.name = f"**{self.name}**\n"

        self.sender = sender
        self.itb = AutoFormattedTB(mode="Plain", color_scheme="NoColor", tb_offset=1)
        get_ipython().set_custom_exc(
            except_tuple, self.custom_exc
        )  # 원하는 Exception만 alert을 받을 수 있습니다

    def send_exception(self, etype, evalue, tb) -> None:
        # TODO: 색상도 알록달록하게 가능하다면 수정
        stb = self.itb.structured_traceback(etype, evalue, tb)
        sstb = self.itb.stb2text(stb)

        self.sender.send(f"{self.name}{sstb}")

    def custom_exc(self, shell, etype, evalue, tb, tb_offset=None) -> None:
        shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)

        self.send_exception(etype, evalue, tb) if self.sender else None
