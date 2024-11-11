
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

// Dot product implementation
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

// Test cases
type V1 = FillArray<1, 1, []>;
type V2 = FillArray<5, 6, []>;

// Should output 130 (1*6 + 2*7 + 3*8 + 4*9 + 5*10)
export type Main<Argv extends string[]> = Print<Dot<V1, V2>>;
