In [1]:
from google.colab import drive
drive.mount('/content/drive/')
Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3aietf%3awg%3aoauth%3a2.0%3aoob&response_type=code&scope=email%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdocs.test%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive%20https%3a%2f%2fwww.googleapis.com%2fauth%2fdrive.photos.readonly%20https%3a%2f%2fwww.googleapis.com%2fauth%2fpeopleapi.readonly

Enter your authorization code:
··········
Mounted at /content/drive/
In [2]:
!pip install selenium
!apt-get -q update # to update ubuntu to correctly run apt install
!apt install -yq chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin

import sys
sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
from selenium import webdriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
Collecting selenium
  Downloading https://files.pythonhosted.org/packages/80/d6/4294f0b4bce4de0abf13e17190289f9d0613b0a44e5dd6a7f5ca98459853/selenium-3.141.0-py2.py3-none-any.whl (904kB)
     |████████████████████████████████| 911kB 10.2MB/s 
Requirement already satisfied: urllib3 in /usr/local/lib/python3.6/dist-packages (from selenium) (1.24.3)
Installing collected packages: selenium
Successfully installed selenium-3.141.0
Get:1 http://security.ubuntu.com/ubuntu bionic-security InRelease [88.7 kB]
Hit:2 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
Hit:3 http://archive.ubuntu.com/ubuntu bionic InRelease
Get:4 http://archive.ubuntu.com/ubuntu bionic-updates InRelease [88.7 kB]
Get:5 http://ppa.launchpad.net/marutter/c2d4u3.5/ubuntu bionic InRelease [15.4 kB]
Get:6 http://archive.ubuntu.com/ubuntu bionic-backports InRelease [74.6 kB]
Ign:7 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
Ign:8 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
Hit:9 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release
Hit:10 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
Get:11 http://security.ubuntu.com/ubuntu bionic-security/main amd64 Packages [739 kB]
Get:12 http://security.ubuntu.com/ubuntu bionic-security/universe amd64 Packages [789 kB]
Get:13 http://ppa.launchpad.net/marutter/c2d4u3.5/ubuntu bionic/main Sources [1,735 kB]
Get:14 http://ppa.launchpad.net/marutter/c2d4u3.5/ubuntu bionic/main amd64 Packages [837 kB]
Get:15 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/ InRelease [3,626 B]
Get:16 http://archive.ubuntu.com/ubuntu bionic-updates/universe amd64 Packages [1,316 kB]
Get:17 http://archive.ubuntu.com/ubuntu bionic-updates/restricted amd64 Packages [29.9 kB]
Get:18 http://archive.ubuntu.com/ubuntu bionic-updates/multiverse amd64 Packages [9,549 B]
Get:19 http://archive.ubuntu.com/ubuntu bionic-updates/main amd64 Packages [1,035 kB]
Fetched 6,762 kB in 2s (2,959 kB/s)
Reading package lists...
Reading package lists...
Building dependency tree...
Reading state information...
The following package was automatically installed and is no longer required:
  libnvidia-common-430
Use 'apt autoremove' to remove it.
The following additional packages will be installed:
  chromium-browser chromium-browser-l10n chromium-codecs-ffmpeg-extra
Suggested packages:
  webaccounts-chromium-extension unity-chromium-extension adobe-flashplugin
The following NEW packages will be installed:
  chromium-browser chromium-browser-l10n chromium-chromedriver
  chromium-codecs-ffmpeg-extra
0 upgraded, 4 newly installed, 0 to remove and 68 not upgraded.
Need to get 71.9 MB of archives.
After this operation, 257 MB of additional disk space will be used.
Get:1 http://archive.ubuntu.com/ubuntu bionic-updates/universe amd64 chromium-codecs-ffmpeg-extra amd64 78.0.3904.108-0ubuntu0.18.04.1 [1,078 kB]
Get:2 http://archive.ubuntu.com/ubuntu bionic-updates/universe amd64 chromium-browser amd64 78.0.3904.108-0ubuntu0.18.04.1 [63.3 MB]
Get:3 http://archive.ubuntu.com/ubuntu bionic-updates/universe amd64 chromium-browser-l10n all 78.0.3904.108-0ubuntu0.18.04.1 [3,076 kB]
Get:4 http://archive.ubuntu.com/ubuntu bionic-updates/universe amd64 chromium-chromedriver amd64 78.0.3904.108-0ubuntu0.18.04.1 [4,466 kB]
Fetched 71.9 MB in 1s (52.9 MB/s)
Selecting previously unselected package chromium-codecs-ffmpeg-extra.
(Reading database ... 145605 files and directories currently installed.)
Preparing to unpack .../chromium-codecs-ffmpeg-extra_78.0.3904.108-0ubuntu0.18.04.1_amd64.deb ...
Unpacking chromium-codecs-ffmpeg-extra (78.0.3904.108-0ubuntu0.18.04.1) ...
Selecting previously unselected package chromium-browser.
Preparing to unpack .../chromium-browser_78.0.3904.108-0ubuntu0.18.04.1_amd64.deb ...
Unpacking chromium-browser (78.0.3904.108-0ubuntu0.18.04.1) ...
Selecting previously unselected package chromium-browser-l10n.
Preparing to unpack .../chromium-browser-l10n_78.0.3904.108-0ubuntu0.18.04.1_all.deb ...
Unpacking chromium-browser-l10n (78.0.3904.108-0ubuntu0.18.04.1) ...
Selecting previously unselected package chromium-chromedriver.
Preparing to unpack .../chromium-chromedriver_78.0.3904.108-0ubuntu0.18.04.1_amd64.deb ...
Unpacking chromium-chromedriver (78.0.3904.108-0ubuntu0.18.04.1) ...
Processing triggers for mime-support (3.60ubuntu1) ...
Setting up chromium-codecs-ffmpeg-extra (78.0.3904.108-0ubuntu0.18.04.1) ...
Processing triggers for man-db (2.8.3-2ubuntu0.1) ...
Processing triggers for hicolor-icon-theme (0.17-2) ...
Setting up chromium-browser (78.0.3904.108-0ubuntu0.18.04.1) ...
update-alternatives: using /usr/bin/chromium-browser to provide /usr/bin/x-www-browser (x-www-browser) in auto mode
update-alternatives: using /usr/bin/chromium-browser to provide /usr/bin/gnome-www-browser (gnome-www-browser) in auto mode
Setting up chromium-chromedriver (78.0.3904.108-0ubuntu0.18.04.1) ...
Setting up chromium-browser-l10n (78.0.3904.108-0ubuntu0.18.04.1) ...
cp: '/usr/lib/chromium-browser/chromedriver' and '/usr/bin/chromedriver' are the same file
Task A: Scraper
In [0]:
import pandas as pd

driver = webdriver.Chrome('chromedriver', options=chrome_options)
In [0]:
df = pd.read_csv("drive/My Drive/Text/Text Project/Links/healthcare in us wordpress google search2.csv")
In [5]:
df['link'][0]
Out[5]:
'https://pmcdeadline2.wordpress.com/2019/09/pod-save-america-producer-crooked-media-to-launch-narrative-podcast-about-healthcare-dr-abdul-el-sayed-1202742106/'
In [0]:
for i in df:
  print(i, df[i])
link 0     https://pmcdeadline2.wordpress.com/2019/09/pod...
1     https://epianalysis.wordpress.com/2012/07/18/u...
2     https://drkevincampbellmd.wordpress.com/2017/0...
3     https://pmcvariety.wordpress.com/2013/tv/news/...
4     https://drkevincampbellmd.wordpress.com/2017/0...
                            ...                        
95    https://stevensonfinancialmarketing.wordpress....
96    https://richardbrenneman.wordpress.com/2010/06...
97    https://mdsaveblog.wordpress.com/2015/06/01/ma...
98    https://homelessphilosopher.wordpress.com/2013...
99    https://ioneblackamericaweb.wordpress.com/2017...
Name: link, Length: 100, dtype: object
In [0]:
import re
data=pd.DataFrame(columns=['link','title','content'])
for i in range(len(df)):
  driver.get(df['link'][i])
  try:
    title_xpath = driver.find_elements_by_xpath('/html/body/div[1]/div[2]/div[1]/div/div[2]/h1')[0]
    title = title_xpath.text
    print('title done')
    content_xpath = driver.find_elements_by_xpath('/html/body/div[1]/div[2]/div[1]/div/div[2]/div[2]')[0]
    content = content_xpath.text
    print('content done')
    data.loc[len(data)]=[df['link'][i], title, content]
  except:
    break
In [7]:
data
Out[7]:
link	title	content
In [0]:
data.to_csv('drive/My Drive/Text/Text Project/Wordpress/us_data.csv')
