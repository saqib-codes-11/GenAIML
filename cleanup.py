from apscheduler.schedulers.background import BackgroundScheduler
import os
import time
from flask import Flask

app = Flask(__name__)

# How often to run cleanup in minutes
CLEANUP_INTERVAL = 10

# How old a fild should be before it is deleted in minutes
CLEANUP_AGE = 10

# Schedule a file cleanup - so that files are removed by the server rather than
# relying on client request to delete the files


def cleanup():
    print("Starting temp file cleanup")
    path = 'static/output'

    listOfFiles = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        # print(dirnames)

    for file in listOfFiles:
        # print(file)
        modification_time = os.path.getmtime(file)
        # print(modification_time)
        #local_time = time.ctime(modification_time)

        #print("Last modification time(Local time):", local_time)
        if((time.time()-modification_time)/60 >= 10):
            # print(file)
            #print("File age in minutes:", (time.time()-modification_time)/60)
            #print("File is older than 10 minutes")
            try:
                os.remove(file)

            except Exception as e:
                print(e)

    # cleanup empty directories - need to cleanup the directories AFTER the files are deleted
    # this function won't delete a non empty directory
    for (dirpath, dirnames, filenames) in os.walk(path):
        for dir in dirnames:
            path = os.path.join(dirpath, dir)

            if os.path.exists(path) and not os.path.isfile(path):
                modification_time = os.path.getmtime(path)
                print(modification_time)
                # Checking if the directory is empty or not AND the folder is old
                if not os.listdir(path) and ((time.time()-modification_time)/60 >= 10):
                    try:
                        os.rmdir(path)
                    except Exception as e:
                        print(e)


# Start  Scheduler to keep file system clean for produced temp files for generated text and audio
sched = BackgroundScheduler(daemon=True)
sched.add_job(cleanup, 'interval', seconds=10)
sched.start()

# Run cleanup on startup to remove old files
cleanup()
