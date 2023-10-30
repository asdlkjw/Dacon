from typing import Any, Dict, List, Tuple
from torch import nn
import torch
import logging
import os
import csv
import pandas as pd
from matplotlib import pyplot as plt


class Recorder:
    def __init__(
        self,
        record_dir: str,
        model: nn.Module = None,
        optimizer: nn.Module = None,
        scheduler: nn.Module = None,
    ) -> None:
        """
        실험 중 epoch의 결과를 기록해주는 클래스
        k-fold의 경우 각 fold별 recorder를 새로 만들어서 기록합니다

        Args:
            record_dir: 결과가 저장될 디렉토리 경로
            model: pytorch model 객체
            optimizer: model과 함께 저장될 optimizer
            scheduler: model과 함께 저장될 scheduler
            plot_dir_name: 시각화해서 만들어질 plot들이 저장될 폴더명
            record_filename: 정보가 기록될 csv 파일명
            model_filename: 모델 weight 파일명
            logger: logger 인스턴스

        """
        self.record_dir = record_dir
        self.plot_dir = os.path.join(record_dir, "plots")
        self.csv_path = os.path.join(record_dir, "record.csv")
        self.current_epoch = 0
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_score = None
        self.row_dict: Dict[str, float] = {}

    def create_record_directory(self) -> None:
        """학습 과정 및 결과가 저장 될 디렉토리 생성"""
        os.makedirs(self.plot_dir, exist_ok=True)

    def is_best_score(self, score: float, what_is_good: str) -> bool:
        """
        현재 score가 가장 좋은 값을 가지고 있는지 비교 후 내부에 저장합니다

        Args:
            score: 비교할 점수 ex) mae loss, accuracy, f1-score..
            what_is_good: 낮은게 좋은지 높은게 좋은지 ('min' or 'max')
        """
        if self.best_score is None:
            self.best_score = score
            return True

        assert what_is_good in ["min", "max"]

        if what_is_good == "min":
            if score < self.best_score:
                self.best_score = score
                return True
        else:
            if score > self.best_score:
                self.best_score = score
                return True

        return False

    def update_row_dict(self, key: str, value: float) -> None:
        """
        csv로 저장할 1행 데이터 정보를 업데이트합니다

        Args:
            key
            value
        """
        self.row_dict[key] = value

    def flush_row_dict(self, is_print=False) -> None:
        """
        현재까지 저장된 row_dict를 csv log로 남깁니다.
        그리고 row_dict는 다시 초기화됩니다

        Outputs:
            recorder/record.csv 에 row_dict 1줄을 남김
        """
        self.create_record_directory()

        fieldnames = list(self.row_dict.keys())

        with open(self.csv_path, newline="", mode="a") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(self.row_dict)

        if is_print == True:
            msg = ""
            for k, v in self.row_dict.items():
                msg += f"{k}: {v}, "
            print(msg[:-2])

        self.row_dict = {}

    def save_checkpoint(
        self, epoch_index: int, checkpoint_filename: str = "checkpoint.pt"
    ) -> None:
        """
        학습 중 체크포인트 가중치 저장
        저장 항목
            - epoch
            - model weight
            - optimizer (optional)
            - scheduler (optional)
            - best_score

        Args:
            epoch_index: epoch_index는 0부터 시작함을 유의
            checkpoint_filename: checkpoint
        """
        self.create_record_directory()

        checkpoint = {
            "epoch": epoch_index + 1,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "best_score": self.best_score,
        }

        save_path = os.path.join(self.record_dir, checkpoint_filename)

        torch.save(checkpoint, save_path)

    def load_checkpoint(
        self,
        device: torch.device,
        checkpoint_path: str = "checkpoint.pt",
    ) -> bool:
        """
        저장한 checkpoint를 불러옵니다

        Args:
            device
            checkpoint_path

        Returns:
            checkpoint load 여부
        """
        if checkpoint_path == "checkpoint.pt":
            checkpoint_path = os.path.join(self.record_dir, checkpoint_path)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.current_epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            ) if self.optimizer else None
            self.scheduler.load_state_dict(
                checkpoint["scheduler"]
            ) if self.scheduler else None
            self.best_score = checkpoint["best_score"]

            return True

        return False

    def save_line_plot(self, plots: List[str], ylim: List = None) -> None:
        """
        record.csv에 있는 train/val을 쌍으로 line plot으로 만들어 저장 (많이 확인하는 자료여서)

        Args:
            plots: record.csv에 저장되는 train_{name}, val_{name} 의 {name}에 들어갈 key 목록
            ylim: plot이 보여질 y 최소, 최대값 입력
        """
        self.create_record_directory()

        record_df = pd.read_csv(self.csv_path)
        max_epoch = record_df["epoch"].max()
        epoch_range = list(range(1, max_epoch + 1))
        color_list = ["red", "blue"]  # train, val

        for plot_name in plots:
            columns = [f"train_{plot_name}", f"val_{plot_name}"]

            fig = plt.figure(figsize=(20, 8))

            for id_, column in enumerate(columns):
                values = record_df[column].tolist()
                plt.plot(
                    epoch_range, values, marker=".", c=color_list[id_], label=column
                )

            plt.title(plot_name, fontsize=15)
            plt.legend(loc="upper right")
            plt.grid()
            plt.xlabel("epoch")
            plt.ylabel(plot_name)
            if ylim is not None:
                plt.ylim(ylim)
            plt.xticks(epoch_range, [str(i) for i in epoch_range])
            plt.close(fig)
            fig.savefig(os.path.join(self.plot_dir, f"{plot_name}.jpg"))
