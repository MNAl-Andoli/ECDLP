# ECDLP

#To run distributed and parallel computing in main PC

pip install -U "ray[default]"

# Install Ray with minimal dependencies
# pip install -U ray


 sudo apt install redis-server
 sudo service redis-server stop


#to Create Head
~/.local/bin/ray start --head --port=6379 --object-store-memory=15000000000 --memory 15000000000

#to create worker
/.local/bin/ray start --address='10.xx.xx.xxx:6379' --redis-password='524159xxxxxxx'
