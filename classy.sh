# Store input arguments: <output_directory> <device> <fp_precision> <input_file>
cd $HOME/mnist
OUTPUT_FILE=.
BATCH_SIZE=$1
DEVICE=$2
#DTYPE=$
FP_MODEL=FP32

# The default path for the job is the user's home directory,
#  change directory to where the files are.

# Make sure that the output directory exists.
#mkdir -p $OUTPUT_FILE

# Check for special setup steps depending upon device to be used
if [ "$DEVICE" = "FPGA" ]; then
    # Environment variables and compilation for edge compute nodes with FPGAs - Updated for OpenVINO 2020.4
    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2
    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-4_PL2_FP16_InceptionV1_ResNet_VGG.aocx
    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

if [ "$BATCH_SIZE" = "100" ]; then
    export MODEL=$HOME/mnist/mnist32.xml
    
else
    export MODEL=$HOME/mnist/1000mnist32.xml
fi

echo "hello"

#make sure you are in the right directory!
python3 $HOME/mnist/classifier.py -m $MODEL -d $DEVICE