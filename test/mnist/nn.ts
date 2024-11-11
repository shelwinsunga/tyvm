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
type DotProduct<
    A extends number[],
    B extends number[],
    Sum extends number = 0
> = A extends [infer X extends number, ...infer RestA extends number[]]
    ? B extends [infer Y extends number, ...infer RestB extends number[]]
        ? DotProduct<RestA, RestB, Add<Sum, Mul<X, Y>>>
        : Sum
    : Sum;

// Test neuron activation with small arrays
type TestInput = [0.5, 0.3, 0.2];
type TestWeights = [0.4, 0.6, 0.2];

// type NeuronActivation<
//     Input extends number[],
//     Weights extends number[],
//     Bias extends number
// > = Sigmoid<Add<DotProduct<Input, Weights>, Bias>>;


type Main<Args extends string[]> = Print<5>;
