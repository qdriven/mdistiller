#!/bin/sh

# Quick training script for optimized CTDKD, GRLCTDKD, and DKD

# Set default values
METHOD="all"  # all now includes DKD
MODEL="res110_res20"
GPU="0"
EPOCHS=300
OUTPUT_DIR="optimized_output"
REPORT_DIR="comparison_report"
ANALYZE=true

# Help message
show_help() {
    echo "Usage: ./quick_train.sh [options]"
    echo "Options:"
    echo "  -m, --method METHOD   Training method: CTDKD, GRLCTDKD, DKD, or all (default: all)"
    echo "  -M, --model MODEL     Model configuration (default: res110_res20)"
    echo "  -g, --gpu GPU         GPU ID to use (default: 0)"
    echo "  -e, --epochs EPOCHS   Number of epochs (default: 300)"
    echo "  -o, --output OUTPUT   Output directory for training (default: optimized_output)"
    echo "  -r, --report REPORT   Output directory for reports (default: comparison_report)"
    echo "  --no-analyze          Skip analysis after training"
    echo "  -h, --help            Show this help message"
}

# Parse arguments
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        -M|--model)
            MODEL="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -r|--report)
            REPORT_DIR="$2"
            shift 2
            ;;
        --no-analyze)
            ANALYZE=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Make sure output and report directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPORT_DIR"

# Process methods based on selection
if [ "$METHOD" = "all" ]; then
    # Can't use arrays in POSIX sh, use space-separated string instead
    METHODS="CTDKD GRLCTDKD DKD"
else
    METHODS="$METHOD"
fi

# Track number of methods
METHOD_COUNT=0

# Train each method
for method in $METHODS; do
    METHOD_COUNT=$((METHOD_COUNT + 1))
    
    echo "============================================================"
    echo "Running training for $method with $MODEL model"
    echo "============================================================"
    
    # Convert method name to lowercase for directory
    method_lower=$(echo "$method" | tr '[:upper:]' '[:lower:]')
    
    # Set method-specific output directory
    method_dir="$OUTPUT_DIR/${method_lower},$MODEL"
    mkdir -p "$method_dir"
    
    # Run training
    CMD="python tools/train.py --cfg configs/cifar100/${method_lower}/$MODEL.yaml"
    # Add overrides
    CMD="$CMD SOLVER.EPOCHS $EPOCHS LOG.PREFIX $method_dir"
    
    # Set GPU
    export CUDA_VISIBLE_DEVICES="$GPU"
    
    # Execute training
    echo "Running: $CMD"
    eval $CMD
    
    # Run analysis if requested
    if [ "$ANALYZE" = true ]; then
        echo "Running analysis for $method..."
        analysis_dir="$REPORT_DIR/${method}_analysis"
        mkdir -p "$analysis_dir"
        
        python tools/monitor_training.py \
            --log_dir "$method_dir" \
            --method "$method" \
            --output_dir "$analysis_dir"
    fi
done

# Run comparison between all methods if more than one
if [ $METHOD_COUNT -gt 1 ]; then
    echo "============================================================"
    echo "Generating comparison report"
    echo "============================================================"
    
    # Create methods arguments string for the comparison command
    METHODS_ARGS=""
    for method in $METHODS; do
        METHODS_ARGS="$METHODS_ARGS $method"
    done
    
    # Create command for comparison
    comp_cmd="python tools/analyze_results.py --logs_dir $OUTPUT_DIR"
    comp_cmd="$comp_cmd --methods $METHODS_ARGS"
    comp_cmd="$comp_cmd --model_name ${MODEL}_comparison"
    comp_cmd="$comp_cmd --model_pattern $MODEL"
    comp_cmd="$comp_cmd --output_dir $REPORT_DIR"
    
    echo "Running: $comp_cmd"
    eval $comp_cmd
    
    echo "============================================================"
    echo "Training and analysis complete!"
    echo "Training outputs: $OUTPUT_DIR"
    echo "Analysis reports: $REPORT_DIR"
    echo "============================================================"
fi 