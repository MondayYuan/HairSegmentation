MODEL="denseunet"
MODEL_PATH="model/denseunet11.pth"
for i in {1..15}
do
    python scripts/main.py --gpu 0 1 --model=${MODEL} --load-path=${MODEL_PATH} test --image=test/test${i}.jpeg --resize=300  --save=test/result${i}.jpg --remove-small-area --unshow
    echo i
done
