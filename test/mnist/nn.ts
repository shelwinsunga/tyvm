// Helper Types
export type Print<T> = T;
export type Rand<Min extends number, Max extends number> = number;
export type Mod<A extends number, B extends number> = number;
export type Add<A extends number, B extends number> = number;
export type Sub<A extends number, B extends number> = number;
export type Mul<A extends number, B extends number> = number;
export type Div<A extends number, B extends number> = number;
export type Exp<A extends number, B extends number> = number;
export type Lte<A extends number, B extends number> = boolean;
export type Lt<A extends number, B extends number> = boolean;
export type Gte<A extends number, B extends number> = boolean;
export type Eq<A extends number, B extends number> = boolean;
export type And<A extends boolean, B extends boolean> = boolean;
export type Or<A extends boolean, B extends boolean> = boolean;

// Increment helper without default parameters
type IncrementHelper<I extends number, Res extends any[]> = 
  Res['length'] extends I
    ? [...Res, any]['length']
    : IncrementHelper<I, [...Res, any]>;

type Increment<I extends number> = IncrementHelper<I, []>;

// Sigmoid function
type E = 2.718281828459045;

type Sigmoid<A extends number> = Div<
  1,
  Add<1, Exp<E, Mul<A, -1>>>
>;

// MakeArray without default parameters
type MakeArrayHelper<N extends number, Result extends number[]> = 
  Result['length'] extends N 
    ? Result
    : MakeArrayHelper<N, [...Result, Rand<0, 1>]>;

type MakeArray<N extends number> = MakeArrayHelper<N, []>;

// MakeMatrix without default parameters
type MakeMatrixHelper<Rows extends number, Cols extends number, Result extends number[][]> = 
  Result['length'] extends Rows 
    ? Result
    : MakeMatrixHelper<Rows, Cols, [...Result, MakeArray<Cols>]>;

type MakeMatrix<Rows extends number, Cols extends number> = MakeMatrixHelper<Rows, Cols, []>;

// Dot product
type DotProduct<
  A extends number[],
  B extends number[]
> = DotProductHelper<A, B, 0, 0>;

type DotProductHelper<
  A extends number[],
  B extends number[],
  Index extends number,
  Accum extends number
> = Index extends A['length']
  ? Accum
  : DotProductHelper<
      A,
      B,
      Increment<Index>,
      Add<Accum, Mul<A[Index], B[Index]>>
    >;

// Neuron activation
type NeuronActivation<
  Input extends number[],
  Weights extends number[],
  Bias extends number
> = Sigmoid<
  Add<
    DotProduct<Input, Weights>,
    Bias
  >
>;

// Hidden layer forward pass with 3 neurons
type HiddenLayerOutput<
  Input extends number[],
  Weights extends number[][],
  Biases extends number[]
> = [
  NeuronActivation<Input, Weights[0], Biases[0]>,
  NeuronActivation<Input, Weights[1], Biases[1]>,
  NeuronActivation<Input, Weights[2], Biases[2]>
];

// Output layer forward pass with 2 neurons
type OutputLayerOutput<
  Input extends number[],
  Weights extends number[][],
  Biases extends number[]
> = [
  NeuronActivation<Input, Weights[0], Biases[0]>,
  NeuronActivation<Input, Weights[1], Biases[1]>
];

// Network parameters
type InputSize = 1;    // MNIST input size
type HiddenSize = 20;    // Adjust as needed
type OutputSize = 10; 

// Generate random input
type Input = MakeArray<InputSize>;

// Generate random weights and biases for hidden layer
type W1 = MakeMatrix<HiddenSize, InputSize>; 
type B1 = MakeArray<HiddenSize>;              

type W2 = MakeMatrix<OutputSize, HiddenSize>; 
type B2 = MakeArray<OutputSize>;              

// Forward pass through hidden layer
type HiddenOutput = HiddenLayerOutput<Input, W1, B1>;

// Forward pass through output layer
type NetworkOutput = OutputLayerOutput<HiddenOutput, W2, B2>;

// Main execution
type Main<Args extends string[]> = Print<NetworkOutput>;