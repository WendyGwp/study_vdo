echo "input orign dataset file path: $1"
echo "input target dataset location: $2"
echo "input target dataset name $3"

OR_PATH=$1
DATA_PATH=$2
DATA_NAME=$3
BEGIN=""
END=""
current_path="$PWD"
cd $DATA_PATH
mkdir $DATA_NAME
cd $DATA_NAME
mkdir image_0
mkdir image_1
mkdir depth
mkdir mask
mkdir flow

python "$current_path"/rename_bagpng.py $OR_PATH

echo "input begin and end split by enter eg 010 119:"
read -p "BEGIN: " BEGIN
read -p "END: " END
python "$current_path"/proessTime.py $OR_PATH $BEGIN $END
sleep 2s
cp "$OR_PATH"/times.txt $DATA_PATH$DATA_NAME
for i in $(seq -w $BEGIN $END); do
    cp "$OR_PATH"image0/0000$i.png "$DATA_PATH$DATA_NAME"/image_0
    cp "$OR_PATH"image1/0000$i.png "$DATA_PATH$DATA_NAME"/image_1
done

sleep 3s

python "$current_path"/rename.py $DATA_PATH$DATA_NAME/image_0 .png ${BEGIN}
python "$current_path"/rename.py $DATA_PATH$DATA_NAME/image_1 .png ${BEGIN}

sleep 3s

echo "processing depth image"
gnome-terminal -t "orign-path" -x bash -c "cd ${DATA_PATH}${DATA_NAME}/depth;${current_path}/spsstereo/build/spsstereo ${DATA_PATH}${DATA_NAME}/image_0/ ${DATA_PATH}${DATA_NAME}/image_1/;exec bash;"

echo "processing flow"
# now have not GPU to process so data from windows
for i in $(seq -w $BEGIN $END); do
    cp "$OR_PATH"flow/000$i.flo "$DATA_PATH$DATA_NAME"/flow
done
python "$current_path"/rename.py $DATA_PATH$DATA_NAME/flow .flo ${BEGIN}

