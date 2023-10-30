## Setup
setup:
```
[git clone https://gitlab.surromind.ai/mlg/dacon_cars.git](https://github.com/asdlkjw/Dacon.git)
cd dacon_cars
python setup.py develop
```

requirements (optional):
```
pip install -r requirements.txt
```

vscode extension install:
```
[Black Formatter]
[Pylance]
```

## 작업 방식
module 같은 경우는
1. 모든 실험에서 공통적으로 사용할 수 있는 것
2. 자주 변경되지 않는 것 (변경되더라도 import 하는 version 경로를 통해 하위호환성을 지켜줍니다)
3. input, output이 변경될 시 version up

실험이 진행되는 .ipynb 파일의 이름을 해당 실험을 식별할 수 있는 id값으로 지어줍니다

실험이 끝난 산출물에는 해당 실험을 재현할 수 있는 .ipynb를 복사해서 같이 보관합니다
