echo '================================================================'
g++ -std=c++11 -shared -Wl,-soname,sim -o sim.so -fPIC sim.cpp 
python sim.py
