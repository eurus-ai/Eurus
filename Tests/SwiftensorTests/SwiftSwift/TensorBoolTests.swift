import XCTest
@testable import Swiftensor

class TensorBoolTests: XCTestCase {
    
    func testBool() {
        let a = Tensor(shape: [2, 2], elements: [true, false, true, false])
        let b = Tensor(shape: [2, 2], elements: [true, true, false, false])
        
        XCTAssertEqual(not(a), Tensor(shape: [2, 2], elements: [false, true, false, true]))
        XCTAssertEqual(and(a, b), Tensor(shape: [2, 2], elements: [true, false, false, false]))
        XCTAssertEqual(or(a, b), Tensor(shape: [2, 2], elements: [true, true, true, false]))
    }
    
    func testCompare() {
        let a = Tensor(shape: [2, 2], elements: [1, 2, 3, 4])
        let b = Tensor(shape: [2, 2], elements: [1, 3, 2, 4])
        
        XCTAssertEqual(equal(a, 2), Tensor(shape: [2, 2], elements: [false, true, false, false]))
        XCTAssertEqual(lessThan(a, 2), Tensor(shape: [2, 2], elements: [true, false, false, false]))
        XCTAssertEqual(greaterThan(a, 3), Tensor(shape: [2, 2], elements: [false, false, false, true]))
        XCTAssertEqual(notGreaterThan(a, 2), Tensor(shape: [2, 2], elements: [true, true, false, false]))
        XCTAssertEqual(notLessThan(a, 3), Tensor(shape: [2, 2], elements: [false, false, true, true]))
        
        XCTAssertEqual(equal(a, b), Tensor(shape: [2, 2], elements: [true, false, false, true]))
        XCTAssertEqual(lessThan(a, b), Tensor(shape: [2, 2], elements: [false, true, false, false]))
        XCTAssertEqual(greaterThan(a, b), Tensor(shape: [2, 2], elements: [false, false, true, false]))
        XCTAssertEqual(notGreaterThan(a, b), Tensor(shape: [2, 2], elements: [true, true, false, true]))
        XCTAssertEqual(notLessThan(a, b), Tensor(shape: [2, 2], elements: [true, false, true, true]))
    }
    
    static var allTests: [(String, (TensorBoolTests) -> () throws -> Void)] {
        return [
            ("testBool", testBool),
            ("testCompare", testCompare)
        ]
    }
}
