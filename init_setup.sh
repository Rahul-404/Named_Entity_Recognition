echo [$(date)]: "START"

echo [$(date)]: "creating env with python 3.9.13 version"

conda create --prefix ./venv python=3.9.13 -y

echo [$(date)]: "activating the environment"

source activate ./venv

echo [$(date)]: "installing the dev requirement"

pip install -r requirements.txt

echo [$(date)]: "END"