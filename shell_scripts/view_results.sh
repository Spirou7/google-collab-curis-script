#!/bin/bash

# Script to help view results from remote SSH server

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_message() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_option() { echo -e "${BLUE}[OPTION $1]${NC} $2"; }
print_warning() { echo -e "${YELLOW}[NOTE]${NC} $1"; }

echo ""
echo "=========================================="
echo "   VIEW RESULTS FROM REMOTE SERVER"
echo "=========================================="
echo ""

# Check if results exist
if [ ! -d "./extracted_results" ]; then
    print_warning "No extracted_results directory found!"
    print_message "Run './shell_scripts/docker_run.sh extract-safe' first"
    exit 1
fi

print_message "Found results in ./extracted_results/"
echo ""
echo "Choose how to view your results:"
echo ""

print_option "1" "Create HTML Report (Best for Remote)"
echo "   Combines all results and images into a single HTML file"
echo "   Command:"
echo "   python create_html_report.py ./extracted_results/optimizer"
echo ""

print_option "2" "Start Web Server (View in Browser)"
echo "   Serves files on a port you can access"
echo "   Commands:"
echo "   cd extracted_results"
echo "   python -m http.server 8888"
echo "   Then visit: http://YOUR_SERVER_IP:8888"
echo ""

print_option "3" "Download Files via SCP"
echo "   Copy files to your local machine"
echo "   From your LOCAL machine, run:"
echo "   scp -r user@server:path/to/extracted_results ./"
echo ""

print_option "4" "Use VS Code Remote SSH"
echo "   Install 'Remote - SSH' extension in VS Code"
echo "   Connect to your server and browse files visually"
echo ""

print_option "5" "Create ZIP for Download"
echo "   Compress everything for easy download"
echo "   Command:"
echo "   zip -r results.zip extracted_results/"
echo ""

echo "=========================================="
echo ""
read -p "Enter option (1-5) or 'q' to quit: " choice

case $choice in
    1)
        print_message "Creating HTML report..."
        python create_html_report.py ./extracted_results/optimizer
        echo ""
        print_message "HTML report created!"
        print_message "Download: extracted_results/optimizer/*/report.html"
        print_message "Open in any web browser to view results and images"
        ;;
    
    2)
        print_message "Starting web server..."
        print_warning "Make sure port 8888 is open in your firewall"
        print_warning "Press Ctrl+C to stop the server"
        echo ""
        cd extracted_results
        echo "Server running at http://$(hostname -I | awk '{print $1}'):8888"
        python -m http.server 8888
        ;;
    
    3)
        print_message "SCP Download Instructions:"
        echo ""
        echo "From your LOCAL machine, run:"
        echo ""
        SERVER_NAME=$(hostname)
        CURRENT_PATH=$(pwd)
        echo "scp -r $USER@$SERVER_NAME:$CURRENT_PATH/extracted_results ./"
        echo ""
        echo "Or for just the latest run:"
        LATEST_RUN=$(ls -t extracted_results/optimizer/ | head -1)
        echo "scp -r $USER@$SERVER_NAME:$CURRENT_PATH/extracted_results/optimizer/$LATEST_RUN ./"
        ;;
    
    4)
        print_message "VS Code Remote SSH Instructions:"
        echo ""
        echo "1. Install VS Code on your local machine"
        echo "2. Install 'Remote - SSH' extension"
        echo "3. Press Ctrl+Shift+P and select 'Remote-SSH: Connect to Host'"
        echo "4. Enter: $USER@$(hostname)"
        echo "5. Navigate to: $(pwd)/extracted_results"
        echo "6. View images directly in VS Code!"
        ;;
    
    5)
        print_message "Creating ZIP archive..."
        zip -r results_$(date +%Y%m%d_%H%M%S).zip extracted_results/
        print_message "Created: results_*.zip"
        echo "Download this file via SCP or SFTP"
        ;;
    
    q|Q)
        echo "Exiting..."
        ;;
    
    *)
        print_warning "Invalid option"
        ;;
esac

echo ""
print_message "Quick tips:"
echo "  • HTML reports are self-contained (images embedded)"
echo "  • Web server option works well for quick viewing"
echo "  • VS Code Remote gives the best experience"
echo "  • ZIP files preserve directory structure"