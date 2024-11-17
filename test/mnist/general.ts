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

// Increment Helper without Default Parameters
type IncrementHelper<I extends number, Acc extends any[]> = 
  Acc['length'] extends I 
    ? [...Acc, any]['length'] 
    : IncrementHelper<I, [...Acc, any]>;

type Increment<I extends number> = IncrementHelper<I, []>;

// Sigmoid Function
type E = 2.718281828459045;

type Sigmoid<A extends number> = Div<
    1,
    Add<1, Exp<E, Mul<A, -1>>>
>;

type MakeArrayHelper<N extends number, Result extends number[]> = 
  Result['length'] extends N 
    ? Result
    : MakeArrayHelper<N, [...Result, Rand<0, 1>]>;

type MakeArray<N extends number> = MakeArrayHelper<N, []>;

type MakeMatrixHelper<Rows extends number, Cols extends number, Result extends number[][]> = 
  Result['length'] extends Rows 
    ? Result
    : MakeMatrixHelper<Rows, Cols, [...Result, MakeArray<Cols>]>;

type MakeMatrix<Rows extends number, Cols extends number> = MakeMatrixHelper<Rows, Cols, []>;

// Dot Product
type Dot<
    A extends number[],
    B extends number[],
> = DotImpl<A, B, 0, 0>;

type DotImpl<
    A extends number[],
    B extends number[],
    I extends number,
    Sum extends number,
> = I extends A['length']
    ? Sum
    : DotImpl<
          A,
          B,
          Add<I, 1>,
          Add<Sum, Mul<A[I], B[I]>>
      >;

// Forward Pass for a Single Neuron
type NeuronForward<
    Input extends number[],
    Weights extends number[],
    Bias extends number
> = Sigmoid<
    Add<
        Dot<Input, Weights>,
        Bias
    >
>;

type LayerForward<
    Input extends number[],
    Weights extends number[][],
    Biases extends number[]
> = LayerForwardImpl<Input, Weights, Biases, 0>;

type LayerForwardImpl<
    Input extends number[],
    Weights extends number[][],
    Biases extends number[],
    Index extends number,
> = Index extends Weights['length']
    ? []
    : [NeuronForward<Input, Weights[Index], Biases[Index]>, ...LayerForwardImpl<Input, Weights, Biases, Add<Index, 1>>];

type InputSize = 64;    
type HiddenSize = 10;   
// type HiddenSize2 = 64;
// type OutputSize = 10;

type image = MakeArray<InputSize>; 
type weights1 = MakeMatrix<HiddenSize, InputSize>;
type biases1 = MakeArray<HiddenSize>;

// type weights2 = MakeMatrix<HiddenSize2, HiddenSize>;
// type biases2 = MakeArray<HiddenSize2>;

// type weights3 = MakeMatrix<OutputSize, HiddenSize2>;
// type biases3 = MakeArray<OutputSize>;

// type image = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
// type weights1 = [[0.1, -0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.3, -0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0], [0.4, -0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]];
// type biases1 = [0.1, 0.1, 0.1, 0.1];

type ForwardResult = LayerForward<image, weights1, biases1>;
// type Forward2Result = LayerForward<ForwardResult, weights2, biases2>;
// type Forward3Result = LayerForward<Forward2Result, weights3, biases3>;

type Main<Args extends string[]> = Print<ForwardResult>;
