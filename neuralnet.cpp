#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <vector>
#include <utility>
#include <time.h>
#include <chrono>
#include <Eigen/Dense>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <matplotlibcpp.h>

using namespace Eigen;
using namespace std;
namespace plt = matplotlibcpp;

typedef chrono::high_resolution_clock Clock;

double randomFunc(double dummy) {
	static boost::mt19937 rng(time(0));
 	static boost::normal_distribution<> nd(0.0,1.0);
  	return nd(rng);
}


typedef double dFunc(double x);
typedef VectorXd vFunc(VectorXd x);

double sigmoid(double z) {
	return 1 / (1 + exp(-z));
}

double sigmoid_deri(double z) {
	return sigmoid(z) * (1 - sigmoid(z));
}

double relu(double z) {
	if (z > 0) return z;
	else return 0;
}

double relu_deri(double z) {
	if (z > 0) return 1;
	else return 0;
}

double leaky_relu(double z) {
	if (z > 0) return z;
	else return 0.01 * z;
}

double leaky_relu_deri(double z) {
	if (z > 0) return 1;
	else return 0.01;
}

double my_tanh(double z) {
	return tanh(z);
}

double my_tanh_deri(double z) {
	return 1 - (tanh(z) * tanh(z));
}


class ActivationFunc {

private:
	dFunc *funcptr;
	dFunc *deriptr;

public:

	ActivationFunc(dFunc &f, dFunc &d) {
		this->funcptr = f;
		this->deriptr = d;
	}

	double func(double x) {
		return (*this->funcptr)(x);
	}

	VectorXd func(VectorXd x) {
		int len = x.size();	
		VectorXd ret(len);
		for (int i = 0; i < len; i++) ret[i] = (*this->funcptr)(x[i]);
		return ret;
	}

	MatrixXd func(MatrixXd x) {
		int len1 = x.innerSize();
		int len2 = x.outerSize();
		MatrixXd ret(len1, len2);
		for (int i = 0; i < len1; i++) {
			for (int j = 0; j < len2; j++) ret(i,j) = (*this->funcptr)(x(i,j));
		}
		return ret;
	}

	double deri(double x) {
		return (*this->deriptr)(x);
	}

	VectorXd deri(VectorXd x) {
		int len = x.size();	
		VectorXd ret(len);
		for (int i = 0; i < len; i++) ret[i] = (*this->deriptr)(x[i]);
		return ret;
	}

	MatrixXd deri(MatrixXd x) {
		int len1 = x.innerSize();
		int len2 = x.outerSize();
		MatrixXd ret(len1, len2);
		for (int i = 0; i < len1; i++) {
			for (int j = 0; j < len2; j++) ret(i, j) = (*this->deriptr)(x(i,j));
		}
		return ret;
	}

};

class Layer {

public:
	int layer_num;
	int layer_size;
	MatrixXd a;
	MatrixXd z;
	VectorXd b;
	MatrixXd w;
	shared_ptr <Layer> prevlayer;
	shared_ptr <ActivationFunc> actfn;

	static int layer_counter;

	// Input layer constructor
	Layer (MatrixXd a) {
		this->layer_num = 0;
		this->layer_size = a.outerSize();
		this->a = a;
		
		this->prevlayer = nullptr;
		this->actfn = nullptr;

		layer_counter++;
	}

	// Non-input layer constructor
	Layer (int layer_size, shared_ptr <Layer> prevlayer, shared_ptr <ActivationFunc> actfn) {

		int m = prevlayer->a.innerSize();

		this->layer_num = layer_counter++;
		this->layer_size = layer_size;
		this->prevlayer = prevlayer;
		this->actfn = actfn;

		//this->w = MatrixXd::Random(layer_size, prevlayer->layer_size);
		//this->b = VectorXd::Random(layer_size);
		this->w = MatrixXd::Zero(layer_size, prevlayer->layer_size).unaryExpr(&randomFunc);
		this->b = VectorXd::Zero(layer_size);

		this->w *= sqrt(double(2) / double((prevlayer->layer_size)));
		//cout<<this->w<<endl;
		
		this->z = MatrixXd::Zero(m, layer_size);
		this->a = MatrixXd::Zero(m, layer_size);

		for (int i = 0; i < m; i++) {
			VectorXd tmp = this->w * prevlayer->a.row(i).transpose() + this->b;
			VectorXd tmp2 = (actfn->func)(tmp);
			this->z.row(i) = tmp;
			this->a.row(i) = tmp2;
		}

		layer_counter++;
	}

	void forward(int lo, int hi) {

		int m = prevlayer->a.innerSize();

		for (int i = lo; i < hi; i++) {
			VectorXd tmp = this->w * prevlayer->a.row(i).transpose() + this->b;
			VectorXd tmp2 = (actfn->func)(tmp);
			this->z.row(i) = tmp;
			this->a.row(i) = tmp2;
		}
	}

};
int Layer::layer_counter = 0;


class Network {

private:
	int num_layers;
	int m;
	MatrixXd samples;
	MatrixXd labels;
	vector <int> layer_sizes;
	vector <shared_ptr <ActivationFunc> > actfns;
	vector <shared_ptr <Layer> > layers;
	double alpha;

public:	

	//@ REQUIRES if n = layer_sizes.size(), then actfns.size() = n + 1
	// because layer_sizes only includes hidden layer sizes but the output layer
	// also has an activation function
	//@ REQUIRES the first dimention of samples and labels must be equal (number of samples = m)
	Network (MatrixXd samples, MatrixXd labels, vector <int> layer_sizes, vector <shared_ptr <ActivationFunc> > actfns, double alpha) {

		// layer_sizes only includes the hidden layer sizes
		this->layer_sizes.push_back(samples.outerSize());
		for (int i = 0; i < layer_sizes.size(); i++) {
			this->layer_sizes.push_back(layer_sizes[i]);
		}
		this->layer_sizes.push_back(labels.outerSize());

		// + 2 because input and output layer
		this->num_layers = layer_sizes.size() + 2;

		this->samples = samples;
		this->labels = labels;
		this->m = samples.innerSize();


		// layers contains n + 2 pointers to layers, including the input and the output layer
		shared_ptr <Layer> il = make_shared <Layer>(samples);

		shared_ptr <Layer> nl;
		layers.push_back(il);
		for (int i = 1; i < num_layers; i++) {
			nl = make_shared <Layer>(this->layer_sizes[i], il, actfns[i-1]);
			layers.push_back(nl);
			il = nl;
		}
		this->alpha = alpha;
	}

	void feedForward(int lo, int hi) {
		for (int i = 1; i < num_layers; i++) layers[i]->forward(lo, hi);
	}

	void backPropagate(int lo, int hi) {

		shared_ptr <Layer> ol = layers[num_layers - 1];
		vector <MatrixXd> deltas(num_layers);
		for (int i = 0; i < num_layers; i++) deltas[i] = MatrixXd::Zero(hi - lo, layers[i]->layer_size);

		// deltas need to be only generated for non-input layers, but our deltas vector has 
		// num_layers elements just to maintain index consistency
		for (int i = lo; i < hi; i++) {
			VectorXd tmp = ol->a.row(i) - labels.row(i);
			VectorXd tmp3 = ol->z.row(i);
			VectorXd tmp2 = ol->actfn->deri(tmp3);
			deltas[num_layers - 1].row(i - lo) = tmp.cwiseProduct(tmp2);
		}

		shared_ptr <Layer> nl;
		for (int i = num_layers - 2; i >= 1; i--) {
			ol = layers[i+1];
			nl = layers[i];
			for (int j = lo; j < hi; j++) {
				VectorXd tmp = deltas[i+1].row(j - lo) * (ol->w);
				VectorXd tmp3 = nl->z.row(j);
				VectorXd tmp2 = ol->actfn->deri(tmp3);
				deltas[i].row(j - lo) = (tmp).cwiseProduct(tmp2);
			}
		}

		for (int i = num_layers - 1; i >= 1; i--) {
			ol = layers[i];
			nl = layers[i-1];

			VectorXd b_delta = VectorXd::Zero(ol->layer_size);
			MatrixXd w_delta = MatrixXd::Zero(ol->layer_size, nl->layer_size);

			for (int j = lo; j < hi; j++) {
				b_delta += deltas[i].row(j - lo);
				//if (i == num_layers-1) cout<<nl->a.row(j)<<endl;
				w_delta += deltas[i].row(j - lo).transpose() * (nl->a.row(j));
			}
			
			ol->b -= (alpha * b_delta) / (hi - lo);
			ol->w -= (alpha * w_delta) / (hi - lo);
		}

	}

	double computeLoss() {

		// output layer
		shared_ptr <Layer> ol = layers[num_layers - 1];

		double loss = 0;
		for (int i = 0; i < this->m; i++) {
			VectorXd tmp = (ol->a.row(i) - labels.row(i));
			loss += tmp.dot(tmp) / 2;
		}
		return loss;
	}

	void run(int epochs, int minibatch_size) {

		vector <double> losses;

		int batches = ceil(double(this->m) / double(minibatch_size));

		for (int e = 0; e < epochs; e++) {

			//double tmpalpha = this->alpha;

			for (int i = 0; i < batches; i++) {

				int lo = minibatch_size * i;
				int hi = min(minibatch_size * (i + 1), this->m);
				
				backPropagate(lo, hi);
				feedForward(lo, hi);

				//this->alpha *= sqrt(double(10));		
			}

			double loss = computeLoss();
			cout<<"Epoch #"<<e<<" --> Total loss: "<<loss<<endl;

			//cout<<layers[num_layers-1]->w<<endl;
			//cout<<layers[num_layers-2]->w<<endl;

			//this->alpha = tmpalpha;

			losses.push_back(loss);	
		}

		plt::plot(losses);
		plt::show();
	}

	MatrixXd predict(MatrixXd x_test) {

		MatrixXd y_test = MatrixXd::Zero(x_test.innerSize(), layers[num_layers-1]->layer_size);

		for (int i = 0; i < x_test.innerSize(); i++) {
			VectorXd input = x_test.row(i);
			for (int j = 1; j < this->num_layers; j++) {
				shared_ptr <Layer> ol = layers[j];
				input = ol->w * input + ol->b;
				input = ol->actfn->func(input);
			}
			y_test.row(i) = input.transpose();
		}

		return y_test;
	}

};

pair<MatrixXd, MatrixXd> readinput(string file, int n) {

	ifstream ifile(file);

	MatrixXd label = MatrixXd::Zero(n, 10);

	MatrixXd sample = MatrixXd::Zero(n, 28 * 28);

	for (int i = 0; i < n; i++) {
		string tmp;
		getline(ifile, tmp, ',');
		
		label(i, stoi(tmp, nullptr)) = 1;
	
		for (int j = 0; j < 28 * 28 - 1; j++) {
			getline(ifile, tmp, ',');
			sample(i, j) = stoi(tmp, nullptr);
			sample(i, j) /= 255;
		}
		getline(ifile, tmp, '\n');
		sample(i, 28 * 28 - 1) = stoi(tmp, nullptr);
		sample(i, 28 * 28 - 1) /= 255;
	}

	return make_pair(sample, label);
}

int main() {

	shared_ptr <ActivationFunc> sigmoid_activation = make_shared <ActivationFunc>(sigmoid, sigmoid_deri);
	shared_ptr <ActivationFunc> relu_activation = make_shared <ActivationFunc>(relu, relu_deri);
	shared_ptr <ActivationFunc> leaky_relu_activation = make_shared <ActivationFunc>(leaky_relu, leaky_relu_deri);
	shared_ptr <ActivationFunc> tanh_activation = make_shared <ActivationFunc>(my_tanh, my_tanh_deri);

	string trainfile("mnist_train.csv");
	string testfile("mnist_test.csv");

	pair<MatrixXd, MatrixXd> tmp = readinput(trainfile, 10000);
	MatrixXd x_train = tmp.first;
	MatrixXd y_train = tmp.second;

	cout<<"Training set read finished"<<endl;
	cout<<x_train.innerSize()<<" training examples"<<endl;

	vector <int> layer_sizes{20,20};
	vector <shared_ptr <ActivationFunc> > actfns{relu_activation, relu_activation, sigmoid_activation};

	auto cstart = Clock::now();
	shared_ptr <Network> network = make_shared <Network>(x_train, y_train, layer_sizes, actfns, 0.001);
	network->run(1000, 10);
	auto cend = Clock::now();
	cout<<"Neural Network training time: "<<chrono::duration_cast<chrono::seconds>(cend-cstart).count()<<endl;

	tmp = readinput(testfile, 10000);
	MatrixXd x_test = tmp.first;
	MatrixXd y_test = tmp.second;

	cout<<"\nTest set read finished"<<endl;
	cout<<x_test.innerSize()<<" test examples"<<endl;

	MatrixXd y = network->predict(x_test);
	int hits = 0;

	for (int i = 0; i < 10000; i++) {
		double maxi = 0;
		int argmaxi = -1;
		for (int j = 0; j < 10; j++) {
			if (y(i,j) > maxi) {
				maxi = y(i,j);
				argmaxi = j;
			}
		}

		for (int j = 0; j < 10; j++) {
			if (y_test(i,j) == 1) {
				if (argmaxi == j) hits++;
				break;
			}
		}
	}

	cout<<"Accuracy on test data: "<<double(hits)/100<<"%"<<endl;

	y = network->predict(x_train);
	hits = 0;

	for (int i = 0; i < 10000; i++) {
		double maxi = 0;
		int argmaxi = -1;
		for (int j = 0; j < 10; j++) {
			if (y(i,j) > maxi) {
				maxi = y(i,j);
				argmaxi = j;
			}
		}

		for (int j = 0; j < 10; j++) {
			if (y_train(i,j) == 1) {
				if (argmaxi == j) hits++;
				break;
			}
		}
	}

	cout<<"Accuracy on training data: "<<double(hits)/100<<"%"<<endl;

	return 0;
}