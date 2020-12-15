import XCTest
@testable import Swiftensor

class SwiftensorTests: XCTestCase {
    func testExample() {
        let x = Tensor.range(from: 0, to: 10, stride: 0.01)
        let y = cos(x/2)
        print(y)
    }
    
    static var allTests: [(String, (SwiftensorTests) -> () throws -> Void)] {
        return [
            ("testExample", testExample)
        ]
    }
}
