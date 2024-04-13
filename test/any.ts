import { Print, Add, Sub, Lte, Eq, Panic, AssertEq } from "./std";

export type Main<Args extends string[]> = AssertEq<
  Print<any extends any ? "yes" : "no">,
  "yes"
>;
