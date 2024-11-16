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
          Increment<I>,
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

// Forward Pass for Entire Layer
type LayerForward<
    Input extends number[],
    Weights extends number[][],
    Biases extends number[]
> = [
    NeuronForward<Input, Weights[0], Biases[0]>,
    NeuronForward<Input, Weights[1], Biases[1]>,
    NeuronForward<Input, Weights[2], Biases[2]>,
    NeuronForward<Input, Weights[3], Biases[3]>
];

// Sample Data
type image = [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

type Weights = [
    [0.1, -0.2,  0.3,  0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.2,  0.3, -0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0,  0.1,  0.2,  0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.4, -0.1,  0.1,  0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
];

type Biases = [0.1, 0.1, 0.1, 0.1];

type ForwardResult = LayerForward<image, Weights, Biases>;

type Main<Args extends string[]> = Print<ForwardResult>;