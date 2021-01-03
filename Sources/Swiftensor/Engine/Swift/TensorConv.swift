
public extension Tensor {
    
}


public func img2col<T:ZeroOne>(value: Tensor<T>, batchSize: Int, channels: Int, height: Int, width: Int, kernelHeight: Int, kernelWidth: Int, padding: Int, stride: Int) -> Tensor<T> {
    
    let resultHeight = (value.shape[2] + 2 * padding - kernelHeight) / stride + 1
    let resultWidth = (value.shape[3] + 2 * padding - kernelWidth) / stride + 1
    
    let resultShape = [
        value.shape[1] * kernelWidth * kernelHeight,
        resultHeight * resultWidth * value.shape[0]
    ]
    let count = resultShape[0]*resultShape[1]
    
    let outPointer = UnsafeMutablePointer<T>.allocate(capacity: count)
    defer { outPointer.deallocate() }
    
    let src = UnsafePointer(value.data)
    let dst = outPointer
    
    let depth_stride = width * height;
    let featuremap_stride = depth_stride * channels;
    
    let output_height = (height + 2 * padding - kernelHeight) / stride + 1;
    let output_width = (width + 2 * padding - kernelWidth) / stride + 1;
    let dst_batch_stride = output_width * output_height;
    let dst_full_stride = dst_batch_stride * batchSize;
    
    for k in 0 ..< kernelWidth &* kernelHeight &* channels {
        let kx = k % kernelWidth
        let kyz = k / kernelWidth
        let ky = kyz % kernelHeight
        let kz = kyz / kernelHeight
        for b in 0 ..< batchSize {
            
            for y in 0 ..< output_height {
                let in_y = y &* stride &- padding &+ ky
                
                if in_y >= 0 && in_y < height {
                    for x in 0 ..< output_width {
                        let in_x = x &* stride &- padding &+ kx
                        let input: T
                        if (in_x >= 0 && in_x < width) {
                            input = src[in_x &+ in_y &* width &+ kz * depth_stride &+ b &* featuremap_stride]
                        } else {
                            input = T.zero
                        }
                        dst[dst_full_stride &* k &+ b &* dst_batch_stride &+ y &* output_width &+ x] = input
                    }
                } else {
                    let dst_float = (dst as! UnsafeMutablePointer<T>)
                    let dst_offset = dst_float.advanced(by: dst_full_stride &* k &+ b &* dst_batch_stride &+ y &* output_width)
                    for i in 0 ..< output_width {
                        dst_offset[i] = T.zero;
                    }
                }
            }
        }
    }
    return Tensor(shape: resultShape,
                   elements: Array(UnsafeBufferPointer(start: outPointer, count: count)))
}
