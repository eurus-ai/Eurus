// MARK: - Operators
public prefix func ! (arg: Tensor<Bool>) -> Tensor<Bool> {
    return not(arg)
}
public func && (lhs: Tensor<Bool>, rhs: Tensor<Bool>) -> Tensor<Bool> {
    return and(lhs, rhs)
}
public func || (lhs: Tensor<Bool>, rhs: Tensor<Bool>) -> Tensor<Bool> {
    return or(lhs, rhs)
}

public func ==<T: Equatable>(lhs: Tensor<T>, rhs: T) -> Tensor<Bool> {
    return equal(lhs, rhs)
}
public func ==<T: Equatable>(lhs: T, rhs: Tensor<T>) -> Tensor<Bool> {
    return equal(rhs, lhs)
}
public func <<T: Comparable>(lhs: Tensor<T>, rhs: T) -> Tensor<Bool> {
    return lessThan(lhs, rhs)
}
public func <<T: Comparable>(lhs: T, rhs: Tensor<T>) -> Tensor<Bool> {
    return greaterThan(rhs, lhs)
}
public func ><T: Comparable>(lhs: Tensor<T>, rhs: T) -> Tensor<Bool> {
    return greaterThan(lhs, rhs)
}
public func ><T: Comparable>(lhs: T, rhs: Tensor<T>) -> Tensor<Bool> {
    return lessThan(rhs, lhs)
}
public func <=<T: Comparable>(lhs: Tensor<T>, rhs: T) -> Tensor<Bool> {
    return notGreaterThan(lhs, rhs)
}
public func <=<T: Comparable>(lhs: T, rhs: Tensor<T>) -> Tensor<Bool> {
    return notLessThan(rhs, lhs)
}
public func >=<T: Comparable>(lhs: Tensor<T>, rhs: T) -> Tensor<Bool> {
    return notLessThan(lhs, rhs)
}
public func >=<T: Comparable>(lhs: T, rhs: Tensor<T>) -> Tensor<Bool> {
    return notGreaterThan(rhs, lhs)
}

public func ==<T: Equatable>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<Bool> {
    return equal(lhs, rhs)
}
public func <<T: Comparable>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<Bool> {
    return lessThan(lhs, rhs)
}
public func ><T: Comparable>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<Bool> {
    return greaterThan(lhs, rhs)
}
public func <=<T: Comparable>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<Bool> {
    return notGreaterThan(lhs, rhs)
}
public func >=<T: Comparable>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<Bool> {
    return notLessThan(lhs, rhs)
}

// MARK: - Unary
func not(_ arg: Tensor<Bool>) -> Tensor<Bool> {
    return apply(arg, !)
}

// MARK: - Binary
func and(_ lhs: Tensor<Bool>, _ rhs: Tensor<Bool>) -> Tensor<Bool> {
    // FIXME: Setting third argument `&&` causes error
    return combine(lhs, rhs) { $0 && $1 }
}

func or(_ lhs: Tensor<Bool>, _ rhs: Tensor<Bool>) -> Tensor<Bool> {
    // FIXME: Setting third argument `||` causes error
    return combine(lhs, rhs) { $0 || $1 }
}

// MARK: - Tensor and scalar
func equal<T: Equatable>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<Bool> {
    return apply(lhs) { $0 == rhs}
}

func lessThan<T: Comparable>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<Bool> {
    return apply(lhs) { $0 < rhs}
}

func greaterThan<T: Comparable>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<Bool> {
    return apply(lhs) { $0 > rhs}
}

func notGreaterThan<T: Comparable>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<Bool> {
    return apply(lhs) { $0 <= rhs}
}

func notLessThan<T: Comparable>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<Bool> {
     return apply(lhs) { $0 >= rhs}
}

// MARK: - Tensor and Tensor
func equal<T: Equatable>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<Bool> {
    return combine(lhs, rhs, ==)
}

func lessThan<T: Comparable>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<Bool> {
    return combine(lhs, rhs, <)
}

func greaterThan<T: Comparable>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<Bool> {
    return combine(lhs, rhs, >)
}

func notGreaterThan<T: Comparable>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<Bool> {
    return combine(lhs, rhs, <=)
}

func notLessThan<T: Comparable>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<Bool> {
    return combine(lhs, rhs, >=)
}
