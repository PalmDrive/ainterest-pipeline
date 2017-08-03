from ailab.tasks.contentType.contentType_class import ContentTypeClass
from ailab.tasks.field.field_class import FieldClass
from ailab.tasks.position.position_class import PositionClass

default_app_config = 'api.apps.ApiConfig'

classifiers = {
    'field': FieldClass('../../ailab/output/field'),
    'position': PositionClass('../../ailab/output/position'),
    'content_type': ContentTypeClass('../../ailab/output/content_type'),
}
