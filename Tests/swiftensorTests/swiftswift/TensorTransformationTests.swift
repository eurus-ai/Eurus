import XCTest
@testable import Swiftensor

class TensorTransformationTests: XCTestCase {
    
    func testReshape() {
        let elements = [Int](0..<12)
        let a = Tensor(shape: [3, 4], elements: elements)
        
        XCTAssertEqual(a.reshaped([6, 2]), Tensor(shape: [6, 2], elements: elements))
        XCTAssertEqual(a.reshaped([3, 2, 2]), Tensor(shape: [3, 2, 2], elements: elements))
        
        // minus
        XCTAssertEqual(a.reshaped([-1, 2]), Tensor(shape: [6, 2], elements: elements))
        XCTAssertEqual(a.reshaped([6, -1]), Tensor(shape: [6, 2], elements: elements))
        XCTAssertEqual(a.reshaped([-1, 2, 2]), Tensor(shape: [3, 2, 2], elements: elements))
        
        var b = Tensor(shape: [3, 4], elements: elements)
        b.shape = [-1, 1]
        XCTAssertEqual(b, Tensor(shape: [12, 1], elements: elements))
        b.shape = [-1]
        XCTAssertEqual(b, Tensor(shape: [12], elements: elements))
    }
    
    func testRavel() {
        let elements = [Int](0..<12)
        let a = Tensor(shape: [3, 4], elements: elements)
        
        XCTAssertEqual(a.raveled(), Tensor(shape: [12], elements: elements))
    }
    
    func testTranspose() {
        let a = Tensor(shape: [2, 3, 4], elements: Array(0..<2*3*4))
        XCTAssertEqual(a.transposed(),
                       Tensor(shape: [4, 3, 2],
                               elements: [0, 12, 4, 16, 8, 20,
                                          1, 13, 5, 17, 9, 21,
                                          2, 14, 6, 18, 10, 22,
                                          3, 15, 7, 19, 11, 23]))
        XCTAssertEqual(a.transposed([0, 2, 1]),
                       Tensor(shape: [2, 4, 3],
                               elements: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11,
                                          12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23]))
        XCTAssertEqual(a.transposed([1, 0, 2]),
                       Tensor(shape: [3, 2, 4],
                               elements: [0, 1, 2, 3,
                                          12, 13, 14, 15,
                                          4, 5, 6, 7,
                                          16, 17, 18, 19,
                                          8, 9, 10, 11,
                                          20, 21, 22, 23]))
        XCTAssertEqual(a.transposed([2, 0, 1]),
                       Tensor(shape: [4, 2, 3],
                               elements: [0, 4, 8, 12, 16, 20,
                                          1, 5, 9, 13, 17, 21,
                                          2, 6, 10, 14, 18, 22,
                                          3, 7, 11, 15, 19, 23]))
    }
    
    static var allTests: [(String, (TensorTransformationTests) -> () throws -> Void)] {
        return [
            ("testReshape", testReshape),
            ("testTranspose", testTranspose)
        ]
    }
}
