
"""
A switching function which interpolates between 0 and max_values with a
cosine function parameterized by min and max coordinates. The
value of the switching function will be between 0 and max_value inside
the min and max limits. max_value is on the left of x_min and 0 on the
right of x_max.
"""
function cosine_switch_off(x::Float64, x_min::Float64, x_max::Float64, max_value::Float64)
	if x <= x_min
		return max_value
	elseif x > x_max
		return 0.0
	end
	return max_value * 0.5 * (1 + cos(π * (x - x_min) / (x_max - x_min)))
end

function cosine_switch_on(x::Float64, x_min::Float64, x_max::Float64, max_value::Float64)
	if x <= x_min
		return 0.0
	elseif x > x_max
		return max_value
	end
	return max_value * (1.0 - 0.5 * (1 + cos(π * (x - x_min) / (x_max - x_min))))
end