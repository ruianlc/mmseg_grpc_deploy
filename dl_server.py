import os
import sys
import grpc
from concurrent import futures
import argparse
import traceback
import multiprocessing

# import dl_b64_pb2
# import dl_b64_pb2_grpc
import dl_abi_pb2
import dl_abi_pb2_grpc


def ip(request):
    print(sys.path)
    import infer_cal
    print(sys.modules['infer_cal'])
    print('开始预测...')
    try:
        result = infer_cal.main(request.model_path, request.config_path, request.img_path)
    except Exception as e:
        raise e

    finally:
        # 预测完成，需要将infer_cal环境变量删除，否则下次import的时候，会导入之前的infer_cal函数
        if 'infer_cal' in sys.modules:
            print('删除环境变量：', sys.modules['infer_cal'])
            sys.modules.pop('infer_cal')
    return result


class Predictor(dl_abi_pb2_grpc.PredictorServicer):

    def predict(self, request, context):
        is_error = False
        error_message = ''
        try:
            print('当前工作路径为:', os.getcwd())

            # 使用进程方式调用函数，防止显存泄露
            pool = multiprocessing.Pool()
            result = pool.apply_async(ip,args=(request,))
            pool.close()
            pool.join()
            result = result.get()

            print('预测完成')
        except Exception as e:
            print(e)
            error_message = traceback.format_exc()
            is_error = True
            result = ''
            print(error_message)

        response = dl_abi_pb2.PredictRespense(redun_id=request.redun_id, is_error=is_error, error_message=error_message, res_g_ratio=result[0], res_y_ratio=result[1], image=result[2])
        return response


def serve(port, max_workers):
    while True:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        dl_abi_pb2_grpc.add_PredictorServicer_to_server(Predictor(), server)
        server.add_insecure_port('[::]:{port}'.format(port=port))
        server.start()

        print(f"Server started, listening on {port}")
        server.wait_for_termination()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--port', type=int, help='port number', required=False, default=50052)
    parser.add_argument('--max_workers', type=int, help='# max workers', required=False, default=10)
    args = parser.parse_args()

    serve(port=args.port, max_workers=args.max_workers)
