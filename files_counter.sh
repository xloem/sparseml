while true
do
    echo "$(date '+TIME:%H:%M:%S') $(awk '{print $1}' /proc/sys/fs/file-nr)" | tee -a logfile
    sleep 2
done