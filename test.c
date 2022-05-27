#include "Tinn.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

// Data object.
typedef struct
{
    // 2D floating point array of input.
    float** in;
    // 2D floating point array of target.
    float** tg;
    // Number of inputs to neural network.
    int nips;
    // Number of outputs to neural network.
    int nops;
    // Number of rows in file (number of sets for neural network).
    int rows;
}
Data;

// Returns the number of lines in a file.
static int lns(FILE* const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while((ch = getc(file)) != EOF)
    {
        if(ch == '\n')
            lines++;
        pc = ch;
    }
    if(pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

// Reads a line from a file.
static char* readln(FILE* const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char* line = (char*) malloc((size) * sizeof(char));
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
            line = (char*) realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

// New 2D array of floats.
static float** new2d(const int rows, const int cols)
{
    float** row = (float**) malloc((rows) * sizeof(float*));
    for(int r = 0; r < rows; r++)
        row[r] = (float*) malloc((cols) * sizeof(float));
    return row;
}

// New data object.
static Data ndata(const int nips, const int nops, const int rows)
{
    const Data data = {
        new2d(rows, nips), new2d(rows, nops), nips, nops, rows
    };
    return data;
}

// Gets one row of inputs and outputs from a string.
static void parse(const Data data, char* line, const int row)
{
    const int cols = data.nips + data.nops;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < data.nips)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.nips] = val;
    }
}

// Frees a data object from the heap.
static void dfree(const Data d)
{
    for(int row = 0; row < d.rows; row++)
    {
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}

// Randomly shuffles a data object.
static void shuffle(const Data d)
{
    for(int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
        float* ot = d.tg[a];
        float* it = d.in[a];
        // Swap output.
        d.tg[a] = d.tg[b];
        d.tg[b] = ot;
        // Swap input.
        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}

// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
static Data build(const char* path, const int nips, const int nops)
{
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        printf("Could not open %s\n", path);
        printf("Get it from the machine learning database: ");
        printf("wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data\n");
        exit(1);
    }
    const int rows = lns(file);
    Data data = ndata(nips, nops, rows);
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

// Learns and predicts hand written digits with 98% accuracy.
int main()
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    // Input and output size is harded coded here as machine learning
    // repositories usually don't include the input and output size in the data itself.
//#define MUL10
#define XOR
#ifdef XOR
    const int nips = 2;
    const int nops = 1;
    const int nhid = 1;
    const int iterations = 100;
    const float anneal = 0.99f;
#elif defined(MUL10)
    const int nips = 1;
    const int nops = 1;
    const int nhid = 28;
    const int iterations = 100;
    const float anneal = 0.99f;
#else
    const int nips = 256;
    const int nops = 10;
    const int nhid = 28;
    const int iterations = 128;
    const float anneal = 0.99f;
#endif
    // Hyper Parameters.
    // Learning rate is annealed and thus not constant.
    // It can be fine tuned along with the number of hidden layers.
    // Feel free to modify the anneal rate.
    // The number of iterations can be changed for stronger training.
    float rate = 1.0f;
    // Load the training set.
#ifdef XOR
    const int datarows = 4;
    Data data = ndata(nips, nops, datarows);
    for (int i = 0; i < datarows; i++) {
        data.in[i] = malloc(nips * sizeof(float));
        for (int j = 0; j < nips; j++) {
            data.in[i][j] = i + 1;
        }
        data.tg[i] = malloc(nops * sizeof(float));
        for (int j = 0; j < nops; j++) {
            data.tg[i][j] = (i + 1) * 10;
        }
    }
    data.in[0][1]=0;data.in[0][0]=0;data.tg[0][0]=0;
    data.in[1][1]=0;data.in[1][0]=1;data.tg[1][0]=0;
    data.in[2][1]=1;data.in[2][0]=0;data.tg[2][0]=0;
    data.in[3][1]=1;data.in[3][0]=1;data.tg[3][0]=1;
#elif defined(MUL10)
    const int datarows = 1000;
    Data data = ndata(nips, nops, datarows);
    for (int i = 0; i < datarows; i++) {
        data.in[i] = malloc(nips * sizeof(float));
        for (int j = 0; j < nips; j++) {
            data.in[i][j] = i + 1;
        }
        data.tg[i] = malloc(nops * sizeof(float));
        for (int j = 0; j < nops; j++) {
            data.tg[i][j] = (i + 1) * 10;
        }
    }
#else
    const Data data = build("semeion.data", nips, nops);
#endif
    // Train, baby, train.
    const Tinn tinn = xtbuild(nips, nhid, nops);
    for(int i = 0; i < iterations; i++)
    {
        shuffle(data);
        float error = 0.0f;
        for(int j = 0; j < data.rows; j++)
        {
            const float* const in = data.in[j];
            const float* const tg = data.tg[j];
            error += xttrain(tinn, in, tg, rate);
        }
        printf("error %.12f :: learning rate %f\n",
            (double) error / data.rows,
            (double) rate);
        rate *= anneal;
    }
    // This is how you save the neural network to disk.
    xtsave(tinn, "saved.tinn");
    xtfree(tinn);
    // This is how you load the neural network from disk.
    const Tinn loaded = xtload("saved.tinn");
    // Now we do a prediction with the neural network we loaded from disk.
    // Ideally, we would also load a testing set to make the prediction with,
    // but for the sake of brevity here we just reuse the training set from earlier.
    // One data set is picked at random (zero index of input and target arrays is enough
    // as they were both shuffled earlier).
    const float* const in = data.in[0];
    const float* const tg = data.tg[0];
    const float* const pd = xtpredict(loaded, in);
    // Prints target.
    xtprint(tg, data.nops);
    // Prints prediction.
    xtprint(pd, data.nops);
    // All done. Let's clean up.
    xtfree(loaded);
    dfree(data);
    return 0;
}
