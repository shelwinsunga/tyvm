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

type MakeWeightArray<Size extends number> = 
    Size extends 1 ? [Rand<-1, 1>] 
    : [...MakeWeightArray<Sub<Size, 1>>, Rand<-1, 1>];

type Main<Args extends string[]> = Print<MakeWeightArray<10>>;
