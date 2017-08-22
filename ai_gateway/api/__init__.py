from ailab.similarity.similarity import Similarity
from ailab.tasks.content_type.content_type_class import ContentTypeClass
from ailab.tasks.field.field_class import FieldClass
from ailab.tasks.position.position_class import PositionClass

default_app_config = 'api.apps.ApiConfig'

classifiers = {
    'field': FieldClass('../ailab/output/field'),
    'position': PositionClass('../ailab/output/position'),
    'content_type': ContentTypeClass('../ailab/output/content_type'),

}

similarity = Similarity(model_dir='../../../output/similarity')
