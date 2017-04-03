from django import template
register = template.Library()

@register.filter
def sort_by(queryset, order):
    return queryset.order_by(order)


@register.filter
def sort_experiment_by_id(queryset):
    return sorted(queryset, key=lambda tup: tup[0].id, reverse=True)
