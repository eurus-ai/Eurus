// MARK: - Unary
public prefix func +<T: SignedNumeric>(arg: Tensor<T>) -> Tensor<T> {
    return unaryPlus(arg)
}

public prefix func -<T: SignedNumeric>(arg: Tensor<T>) -> Tensor<T> {
    return unaryMinus(arg)
}

// MARK: - Tensor and scalar
public func +<T: Arithmetic>(lhs: Tensor<T>, rhs: T) -> Tensor<T> {
    return add(lhs, rhs)
}

public func -<T: Arithmetic>(lhs: Tensor<T>, rhs: T) -> Tensor<T> {
    return subtract(lhs, rhs)
}

public func *<T: Arithmetic>(lhs: Tensor<T>, rhs: T) -> Tensor<T> {
    return multiply(lhs, rhs)
}

public func /<T: Arithmetic>(lhs: Tensor<T>, rhs: T) -> Tensor<T> {
    return divide(lhs, rhs)
}

public func %<T: Moduloable>(lhs: Tensor<T>, rhs: T) -> Tensor<T> {
    return modulo(lhs, rhs)
}

public func +<T: Arithmetic>(lhs: T, rhs: Tensor<T>) -> Tensor<T> {
    return add(lhs, rhs)
}

public func -<T: Arithmetic>(lhs: T, rhs: Tensor<T>) -> Tensor<T> {
    return subtract(lhs, rhs)
}

public func *<T: Arithmetic>(lhs: T, rhs: Tensor<T>) -> Tensor<T> {
    return multiply(lhs, rhs)
}

public func /<T: Arithmetic>(lhs: T, rhs: Tensor<T>) -> Tensor<T> {
    return divide(lhs, rhs)
}

public func %<T: Moduloable>(lhs: T, rhs: Tensor<T>) -> Tensor<T> {
    return modulo(lhs, rhs)
}

// MARK: - Tensor and Tensor
public func +<T: Arithmetic>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<T> {
    return add(lhs, rhs)
}

public func -<T: Arithmetic>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<T> {
    return subtract(lhs, rhs)
}

public func *<T: Arithmetic>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<T> {
    return multiply(lhs, rhs)
}

public func /<T: Arithmetic>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<T> {
    return divide(lhs, rhs)
}

public func %<T: Moduloable>(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<T> {
    return modulo(lhs, rhs)
}

// MARK: - Unary
func unaryPlus<T: SignedNumeric>(_ arg: Tensor<T>) -> Tensor<T> {
    return apply(arg, +)
}

func unaryMinus<T: SignedNumeric>(_ arg: Tensor<T>) -> Tensor<T> {
    return apply(arg, -)
}

// MARK: - Tensor and scalar
func add<T: Arithmetic>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> {
    return apply(lhs) { $0 + rhs }
}

func add<T: Arithmetic>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> {
    return apply(rhs) { lhs + $0 }
}

func subtract<T: Arithmetic>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> {
    return apply(lhs) { $0 - rhs }
}

func subtract<T: Arithmetic>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> {
    return apply(rhs) { lhs - $0 }}

func multiply<T: Arithmetic>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> {
    return apply(lhs) { $0 * rhs }
}

func multiply<T: Arithmetic>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> {
    return apply(rhs) { lhs * $0 }
}

func divide<T: Arithmetic>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> {
    return apply(lhs) { $0 / rhs }
}

func divide<T: Arithmetic>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> {
    return apply(rhs) { lhs / $0 }
}

func modulo<T: Moduloable>(_ lhs: Tensor<T>, _ rhs: T) -> Tensor<T> {
    return apply(lhs) { $0 % rhs }
}

func modulo<T: Moduloable>(_ lhs: T, _ rhs: Tensor<T>) -> Tensor<T> {
    return apply(rhs) { lhs % $0 }
}

// MARK: - Tensor and Tensor
func add<T: Arithmetic>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    return combine(lhs, rhs, +)
}

func subtract<T: Arithmetic>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    return combine(lhs, rhs, -)
}

func multiply<T: Arithmetic>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    return combine(lhs, rhs, *)
}

func divide<T: Arithmetic>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    return combine(lhs, rhs, /)
}

func modulo<T: Moduloable>(_ lhs: Tensor<T>, _ rhs: Tensor<T>) -> Tensor<T> {
    return combine(lhs, rhs, %)
}

// MARK: - Scalar
public func +=<T: Arithmetic>(lhs: inout Tensor<T>, rhs: T) {
    lhs = lhs + rhs
}

public func -=<T: Arithmetic>(lhs: inout Tensor<T>, rhs: T) {
    lhs = lhs - rhs
}

public func *=<T: Arithmetic>(lhs: inout Tensor<T>, rhs: T) {
    lhs = lhs * rhs
}

public func /=<T: Arithmetic>(lhs: inout Tensor<T>, rhs: T) {
    lhs = lhs / rhs
}

public func %=<T: Moduloable>(lhs: inout Tensor<T>, rhs: T) {
    lhs = lhs % rhs
}

// MARK: - Tensor
public func +=<T: Arithmetic>(lhs: inout Tensor<T>, rhs: Tensor<T>) {
    lhs = lhs + rhs
}

public func -=<T: Arithmetic>(lhs: inout Tensor<T>, rhs: Tensor<T>) {
    lhs = lhs - rhs
}

public func *=<T: Arithmetic>(lhs: inout Tensor<T>, rhs: Tensor<T>) {
    lhs = lhs * rhs
}

public func /=<T: Arithmetic>(lhs: inout Tensor<T>, rhs: Tensor<T>) {
    lhs = lhs / rhs
}

public func %=<T: Moduloable>(lhs: inout Tensor<T>, rhs: Tensor<T>) {
    lhs = lhs % rhs
}
