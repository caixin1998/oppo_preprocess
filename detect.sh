THREAD_NUM=8

i=0
path=/home1/caixin/GazeData/GazeCapture
array=$(ls $path)

mkfifo fifo
exec 9<>fifo
rm -f fifo

for ((i=0;i<$THREAD_NUM;i++))
do
    echo >&9
done
i=0
for arg in ${array}; do
  i=`expr $i + 1`
  read -u9
  {
     echo $arg
     echo $i
     CUDA_VISIBLE_DEVICES=`expr $i % 4` python face_detection.py -i /home1/caixin/GazeData/GazeCapture/$arg 
     sleep 1
     echo >&9
  }&
done
wait
echo "\n全部任务执行结束"
