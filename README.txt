*****
See capstoneReportUPDATED.pdf for an overview of the project and instructions on running the code.
*****
And for background, see the project's introductory blogpost: https://medium.com/@jameslucasbaker/machine-learning-versus-the-news-3b5b479d8e6a
*****

The main code can be run directly from the two Jupyter notebooks - each dealing with a distinct step in the end-to-end process. (JamesBakerCapstonePartOne.ipynb and JamesBakerCapstonePartTwo.ipynb)


Dependencies and environment details are as follows:

python3

Stanford CoreNLP from https://stanfordnlp.github.io/CoreNLP/ (see project Report for details)
Java 8 or newer (for Stanford CoreNLP)
Google Cloud Platform (see project Report for details)

conda create -n JLBCapstone python=3.6 anaconda
source activate JLBCapstone
conda install -n JLBCapstone nltk
conda install -n JLBCapstone bokeh
conda install -n JLBCapstone argparse
pip install --upgrade google-cloud-language
conda install -n JLBCapstone spacy
conda install -n JLBCapstone nltk
pip install pycorenlp

python -m spacy download en

sklearn
pandas
numpy
csv
operator
random
statistics



# packages in environment at /Users/jamesbaker/anaconda3/envs/CapstonePython3:
#
# Name                    Version                   Build  Channel
appnope                   0.1.0                    py36_0    conda-forge
asn1crypto                0.24.0                   py36_3    conda-forge
backcall                  0.1.0                      py_0    conda-forge
blas                      1.0                         mkl  
blinker                   1.4                        py_1    conda-forge
bokeh                     0.13.0                   py36_0    conda-forge
boto                      2.49.0                   py36_0  
boto3                     1.8.3                      py_0    conda-forge
botocore                  1.11.4                     py_0    conda-forge
bz2file                   0.98                     py36_0  
bzip2                     1.0.6                         1    conda-forge
ca-certificates           2018.8.24            ha4d7672_0    conda-forge
cachetools                2.1.0                     <pip>
certifi                   2018.8.24                py36_1    conda-forge
cffi                      1.11.5           py36h5e8e0c9_1    conda-forge
chardet                   3.0.4                     <pip>
chardet                   3.0.4                    py36_3    conda-forge
clangdev                  6.0.1                 default_1    conda-forge
cryptography              2.3.1            py36hdffb7b8_0    conda-forge
cryptography-vectors      2.3.1                    py36_0    conda-forge
cycler                    0.10.0                     py_1    conda-forge
cymem                     1.31.2           py36hfc679d8_0    conda-forge
cytoolz                   0.9.0.1          py36h470a237_0    conda-forge
decorator                 4.3.0                      py_0    conda-forge
dill                      0.2.8.2                  py36_0    conda-forge
docutils                  0.14                     py36_1    conda-forge
en-core-web-sm            2.0.0                     <pip>
freetype                  2.9.1                h6debe1e_1    conda-forge
gensim                    3.4.0            py36h1de35cc_0  
google-api-core           1.3.0                     <pip>
google-auth               1.5.1                     <pip>
google-cloud-language     1.0.2                     <pip>
googleapis-common-protos  1.5.3                     <pip>
grpcio                    1.14.2                    <pip>
icu                       58.2                 hfc679d8_0    conda-forge
idna                      2.7                      py36_2    conda-forge
idna                      2.7                       <pip>
intel-openmp              2018.0.3                      0  
ipykernel                 4.9.0                    py36_0    conda-forge
ipython                   6.5.0                    py36_0    conda-forge
ipython_genutils          0.2.0                      py_1    conda-forge
jedi                      0.12.1                   py36_0    conda-forge
jinja2                    2.10                       py_1    conda-forge
jmespath                  0.9.3                      py_1    conda-forge
jupyter_client            5.2.3                      py_1    conda-forge
jupyter_core              4.4.0                      py_0    conda-forge
kiwisolver                1.0.1            py36h2d50403_2    conda-forge
libcxx                    6.0.1                         0    conda-forge
libffi                    3.2.1                hfc679d8_4    conda-forge
libgfortran               3.0.1                h93005f0_2  
libiconv                  1.15                 h470a237_2    conda-forge
libopenblas               0.2.20               hdc02c5d_7  
libpng                    1.6.35               ha92aebf_0    conda-forge
libsodium                 1.0.16               h470a237_1    conda-forge
libxml2                   2.9.8                h422b904_3    conda-forge
llvm-meta                 6.0.1                         0    conda-forge
llvmdev                   6.0.1                hf8ce74a_2    conda-forge
markupsafe                1.0              py36h470a237_1    conda-forge
matplotlib                2.2.3            py36h0e0179f_0    conda-forge
mkl                       2018.0.3                      1  
mkl_fft                   1.0.6                    py36_0    conda-forge
mkl_random                1.0.1                    py36_0    conda-forge
msgpack-numpy             0.4.3.1                    py_0    conda-forge
msgpack-python            0.5.6            py36h2d50403_2    conda-forge
murmurhash                0.28.0           py36hfc679d8_0    conda-forge
ncurses                   6.1                  hfc679d8_1    conda-forge
nltk                      3.2.5                      py_0    conda-forge
numpy                     1.15.1           py36h6a91979_0  
numpy-base                1.15.1           py36h8a80b8c_0  
oauthlib                  2.1.0                      py_0    conda-forge
openssl                   1.0.2p               h470a237_0    conda-forge
packaging                 17.1                       py_0    conda-forge
pandas                    0.23.4           py36hf8a1672_0    conda-forge
parso                     0.3.1                      py_0    conda-forge
pexpect                   4.6.0                    py36_0    conda-forge
pickleshare               0.7.4                    py36_0    conda-forge
pip                       18.0                     py36_1    conda-forge
plac                      0.9.6                      py_1    conda-forge
preshed                   1.0.1            py36hfc679d8_0    conda-forge
prompt_toolkit            1.0.15                     py_1    conda-forge
protobuf                  3.6.1                     <pip>
ptyprocess                0.6.0                    py36_0    conda-forge
pyasn1                    0.4.4                     <pip>
pyasn1-modules            0.2.2                     <pip>
pycorenlp                 0.3.0                     <pip>
pycparser                 2.18                       py_1    conda-forge
pygments                  2.2.0                      py_1    conda-forge
pyjwt                     1.6.4                      py_0    conda-forge
pyopenssl                 18.0.0                   py36_0    conda-forge
pyparsing                 2.2.0                      py_1    conda-forge
pysocks                   1.6.8                    py36_2    conda-forge
python                    3.6.6                h5001a0f_0    conda-forge
python-crfsuite           0.9.6            py36h470a237_0    conda-forge
python-dateutil           2.7.3                      py_0    conda-forge
pytz                      2018.5                     py_0    conda-forge
pyyaml                    3.13             py36h470a237_1    conda-forge
pyzmq                     17.1.2           py36hae99301_0    conda-forge
readline                  7.0                  haf1bffa_1    conda-forge
regex                     2017.11.09               py36_0    conda-forge
requests                  2.19.1                    <pip>
requests                  2.19.1                   py36_1    conda-forge
requests-oauthlib         1.0.0                      py_1    conda-forge
rsa                       3.4.2                     <pip>
s3transfer                0.1.13                   py36_0    conda-forge
scikit-learn              0.19.0           py36h4cafacf_2  
scipy                     0.19.1           py36h3e758e1_3  
setuptools                40.2.0                   py36_0    conda-forge
simplegeneric             0.8.1                      py_1    conda-forge
six                       1.11.0                   py36_1    conda-forge
smart_open                1.6.0                      py_1    conda-forge
spacy                     2.0.12           py36hf8a1672_0    conda-forge
sqlite                    3.24.0               h2f33b56_0    conda-forge
termcolor                 1.1.0                      py_2    conda-forge
thinc                     6.10.3           py36hf8a1672_3    conda-forge
tk                        8.6.8                         0    conda-forge
toolz                     0.9.0                      py_0    conda-forge
tornado                   5.1              py36h470a237_1    conda-forge
tqdm                      4.24.0                     py_1    conda-forge
traitlets                 4.3.2                    py36_0    conda-forge
twython                   3.7.0                      py_0    conda-forge
ujson                     1.35             py36h470a237_1    conda-forge
urllib3                   1.23                     py36_1    conda-forge
wcwidth                   0.1.7                      py_1    conda-forge
wheel                     0.31.1                   py36_1    conda-forge
wrapt                     1.10.11                  py36_0    conda-forge
xz                        5.2.4                h470a237_1    conda-forge
yaml                      0.1.7                h470a237_1    conda-forge
zeromq                    4.2.5                hfc679d8_5    conda-forge
zlib                      1.2.11               h470a237_3    conda-forge
