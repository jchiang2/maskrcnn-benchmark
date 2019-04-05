import logging

from .my_eval import do_my_evaluation


def my_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("my evaluation doesn't support box_only, ignored.")
    logger.info("performing my evaluation, ignored iou_types.")
    return do_my_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
