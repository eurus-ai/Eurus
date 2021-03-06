// swift-tools-version:5.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Eurus",
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "Eurus",
            targets: ["Eurus"]),
        .library(
            name: "Swiftensor",
            targets: ["Swiftensor"]
        ),
        .executable(name: "Examples", targets: ["Examples"])
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/apple/swift-log.git",from: "1.0.0"),
        .package(url: "https://github.com/eurus-ai/SwiftCV.git",from: "0.1.0")
    ],
    targets: [
        // Targets are the example building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "Eurus",
            dependencies: ["Swiftensor",.product(name: "Logging", package: "swift-log")],
            path: "./Sources/Eurus"
        ),
        .target(
            name: "Examples",
            dependencies: ["Eurus","Swiftensor"],
            path: "./Sources/Examples"
        ),
        .target(
            name: "Swiftensor",
            dependencies: ["SwiftCV"],
            path: "./Sources/Swiftensor"
        ),
        .testTarget(
            name: "EurusTests",
            dependencies: ["Eurus"],
            resources: [.copy("Data/Mnist/t10k-images.idx3-ubyte")
                        ,.copy("Data/Mnist/t10k-labels.idx1-ubyte")
                        ,.copy("Data/Mnist/train-images.idx3-ubyte")
                        ,.copy("Data/Mnist/train-labels.idx1-ubyte")]),
        .testTarget(
            name: "SwiftensorTests",
            dependencies: ["Swiftensor"],
            resources: [.copy("SwiftSwift/Lenna.png")]),
        
    ]
)
