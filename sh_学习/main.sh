echo "Start-up system"
gnome-terminal --geometry 10x10+0+0 -- sh open_roscore.sh
gnome-terminal --geometry 10x10+0+0 -- sh open_rosbag.sh
gnome-terminal --geometry 10x10+0+0 -- sh open_rviz.sh
echo "Complete the start"