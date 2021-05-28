# Fighting

## install 

sudo apt-get install openjdk-8-jre xvfb

pip install gym py4j port_for opencv-python

cd FTG/malib && pip install -e .

## Command

所有命令需要在 FTG 文件夹下运行

### Test env

xvfb-run -s "-screen 0 600x400x24" python malib/example/test_env.py