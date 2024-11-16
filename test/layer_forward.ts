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

type E = 2.718281828459045;

type Sigmoid<A extends number> = Div<
    1,
    Add<1, Exp<E, Mul<A, -1>>>
>;

type FillArray<
  Count extends number,
  Value,
  Array extends any[]
> = FillArrayImpl<Count, 0, Value, Array>;

type FillArrayImpl<
  Count extends number,
  I extends number,
  Value,
  Array extends any[]
> = I extends Count
  ? Array
  : FillArrayImpl<Count, Add<I, 1>, Value, [Value, ...Array]>;


// Original array builder for smaller chunks
type MakeArray<N extends number> = N extends 1 ? [Rand<0, 1>] : MakeArrayImpl<N, 0, []>;

// tail recursion = good, this avoids deep recursion where the stack overflows
type MakeArrayImpl<
  N extends number,
  I extends number,
  Acc extends number[]
> = N extends I ? [...Acc, Rand<0, 1>] : MakeArrayImpl<N, Add<I, 1>, [...Acc, Rand<0, 1>]>;

// Compute dot product recursively
// dot product is the sum of the products of the corresponding elements of the two sequences of numbers.
type Dot<
    A extends number[],
    B extends number[],
> = DotImpl<A, B, 0, 0>

type DotImpl<
    A extends number[],
    B extends number[],
    I extends number,
    Sum extends number,
> = I extends A["length"]
    ? Sum
    : DotImpl<A, B, Add<I, 1>, Add<Sum, Mul<A[I], B[I]>>>

    // Test with 2 inputs -> 3 neurons
type Input = [0.5, 0.8];
type Weights = [
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6]
];
type Biases = [0.1, 0.2, 0.3];

// Layer forward with multiple neurons
type LayerForward<
    Input extends number[],
    Weights extends number[][],
    Biases extends number[]
> = LayerForwardImpl<Input, Weights, Biases, 0>;

type LayerForwardImpl<
    Input extends number[],
    Weights extends number[][],
    Biases extends number[],
    I extends number
> = I extends Weights["length"]
    ? []
    : [
        Sigmoid<Add<Dot<Input, Weights[I]>, Biases[I]>>,
        ...LayerForwardImpl<Input, Weights, Biases, Add<I, 1>>
    ];

type Main<Args extends string[]> = Print<
    LayerForward<Input, Weights, Biases>
>;