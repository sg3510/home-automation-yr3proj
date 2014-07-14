import time
import urllib2

def do_something():
    urllib2.urlopen("http://ee-ug1.ee.ic.ac.uk/actual_web2/poll_API.php")

def run():
    while True:
        do_something()
        time.sleep(300)

if __name__ == "__main__":
    run()