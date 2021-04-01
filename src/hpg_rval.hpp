#pragma once
#include "hpg_config.hpp"
#include "hpg_error.hpp"
#include "hpg_export.h"

#include <tuple>
#if HPG_API >= 17
# include <variant>
#endif

namespace hpg {

/** type for containing a value or an error
 *
 * normally appears as a return value type for a method or function that can
 * fail
 */
#if HPG_API >= 17
template <typename T>
using rval_t = std::variant<Error, T>;
#else // HPG_API < 17
template <typename T>
using rval_t = std::tuple<std::unique_ptr<Error>, T>;
#endif // HPG_API >= 17

/** query whether rval_t value contains an error
 */
template <typename T>
HPG_EXPORT inline bool
is_error(const rval_t<T>& rv) {
#if HPG_API >= 17
  return std::holds_alternative<Error>(rv);
#else // HPG_API < 17
  return bool(std::get<0>(rv));
#endif // HPG_API >= 17
}

/** type trait to get type parameter of a template type
 */
template <typename T>
struct HPG_EXPORT rval_value {
  using type = void; /**< parameter type */
};
/** specialized type trait to get value type from rval_t type
 */
template <typename T>
struct rval_value<rval_t<T>> {
  using type = T; /**< parameter type */
};

/** query whether rval_t value contains a (non-error) value
 */
template <typename T>
HPG_EXPORT inline bool
is_value(const rval_t<T>& rv) {
#if HPG_API >= 17
  return std::holds_alternative<T>(rv);
#else // HPG_API < 17
  return !bool(std::get<0>(rv));
#endif // HPG_API >= 17
}

/** get value from an rval_t value
 */
#if __cplusplus >= 201402L
template <typename RV>
HPG_EXPORT inline auto
get_value(RV&& rv) {
  return std::get<1>(std::forward<RV>(rv));
}
#else // __cplusplus < 201402L
template <typename RV>
HPG_EXPORT inline const typename rval_value<RV>::type&
get_value(const RV& rv) {
  return std::get<1>(rv);
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&
get_value(RV& rv) {
  return std::get<1>(rv);
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&&
get_value(RV&& rv) {
  return std::get<1>(std::move(rv));
}
#endif // __cplusplus >= 201402L

/** get error from an rval_t value
 */
#if __cplusplus >= 201402L
template <typename RV>
HPG_EXPORT inline auto
get_error(RV&& rv) {
# if HPG_API >= 17
  return std::get<0>(std::forward<RV>(rv));
# else
  return *std::get<0>(std::forward<RV>(rv));
# endif
}
#else // __cplusplus < 201402L
template <typename RV>
HPG_EXPORT inline const typename rval_value<RV>::type&
get_error(const RV& rv) {
# if HPG_API >= 17
#  error "Unsupported c++ standard and HPG API version"
# else // HPG_API < 17
  return *std::get<0>(rv);
# endif // HPG_API >= 17
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&
get_error(RV& rv) {
# if HPG_API >= 17
#  error "Unsupported c++ standard and HPG API version"
# else // HPG_API < 17
  return *std::get<0>(rv);
# endif // HPG_API >= 17
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&&
get_error(RV&& rv) {
  return *std::get<0>(std::move(rv));
}
#endif // __cplusplus >= 201402L

/** create an rval_t value from a (non-error) value
 */
template <typename T>
HPG_EXPORT inline rval_t<T>
rval(T&& t) {
#if HPG_API >= 17
  return rval_t<T>(std::forward<T>(t));
#else // HPG_API < 17
  return {std::unique_ptr<Error>(), std::forward<T>(t)};
#endif // HPG_API >= 17
}

/** create an rval_t value from an Error
 */
template <typename T>
HPG_EXPORT inline rval_t<T>
rval(const Error& err) {
#if HPG_API >= 17
  return rval_t<T>(err);
#else // HPG_API < 17
  return {std::unique_ptr<Error>(new Error(err)), T()};
#endif // HPG_API >= 17
}

/** apply function that returns a plain value to an rval_t
 */
#if __cplusplus >= 201402L
template <typename RV, typename F>
HPG_EXPORT auto
map(RV&& rv, F f) {
#if HPG_API >= 17
  using T = std::invoke_result_t<F, typename rval_value<RV>::type>;
#else // HPG_API < 17
  using T = typename std::result_of<F(typename rval_value<RV>::type)>::type;
#endif // HPG_API >= 17
  if (is_value(rv))
    return rval<T>(f(get_value(std::forward<RV>(rv))));
  else
    return rval<T>(get_error(std::forward<RV>(rv)));
}

/** apply function that returns an rval_t value to an rval_t
 */
template <typename RV, typename F>
HPG_EXPORT auto
flatmap(RV&& rv, F f) {
#if HPG_API >= 17
  using T =
    typename
    rval_value<std::invoke_result_t<F, typename rval_value<RV>::type>>::type;
#else // HPG_API < 17
  using T =
    typename rval_value<
      typename std::result_of<F(typename rval_value<RV>::type)>::type>::type;
#endif // HPG_API >= 17

  if (is_value(rv))
    return f(get_value(std::forward<RV>(rv)));
  else
    return rval<T>(get_error(std::forward<RV>(rv)));
}

/** apply function depending on contained value type with common result type
 */
template <typename RV, typename ValF, typename ErrF>
HPG_EXPORT auto
fold(RV&& rv, ValF vf, ErrF ef) {
#if HPG_API >= 17
  static_assert(
    std::is_same_v<
      std::invoke_result_t<ValF, typename rval_value<RV>::type>,
      std::invoke_result_t<ErrF, Error>>);
#else // hpg_api < 17
  static_assert(
    std::is_same<
      typename std::result_of<ValF(typename rval_value<RV>::type)>::type,
      typename std::result_of<ErrF(Error)>::type>::value);
#endif // hpg_api >= 17

  if (is_value(rv))
    return vf(get_value(std::forward<RV>(rv)));
  else
    return ef(get_error(std::forward<RV>(rv)));
}
#endif // __cplusplus >= 201402L

#if HPG_API >= 17 || defined(HPG_INTERNAL)
// The following classes may be of use in a functional programming style of
// error handling; otherwise, they can be safely ignored.

/** monadic class of functions with domain A and range rval_t\<B> */
template <typename A, typename B>
struct RvalM;

/** subclass of RvalM<A> with explicit function type */
template <typename A, typename B, typename F>
struct RvalMF;

template <typename A, typename B>
struct RvalM {

  /** construct an RvalM value as RvalMF<A,F> value with deduced function
   * type */
  template <typename F>
  static RvalMF<A, B, F>
  pure(F&& f) {
    return RvalMF<A, B, F>(std::forward<F>(f));
  }
};

template <typename A, typename B, typename F>
struct RvalMF :
  public RvalM<A, B> {

  static_assert(
    std::is_same_v<B, typename rval_value<std::invoke_result_t<F, A>>::type>);

  /** value type of G applied to value of type B */
  template <typename G>
  using gv_t = typename rval_value<std::invoke_result_t<G, B>>::type;

  /** value type of G applied to a loop index and value of type B */
  template <typename G>
  using giv_t =
    typename rval_value<std::invoke_result_t<G, unsigned, B>>::type;

  F m_f; /**< wrapped function, as value */

  /** constructor */
  RvalMF(const F& f)
    : m_f(f) {}

  /** constructor */
  RvalMF(F&& f)
    : m_f(std::move(f)) {}

  /** apply contained function to a value */
  rval_t<B>
  operator()(A&& a) const {
    return m_f(std::forward<A>(a));
  }

  /** composition with rval_t-valued function */
  template <typename G>
  auto
  and_then(const G& g) const {

    return
      RvalM<A, gv_t<G>>::pure(
        [g, f=m_f](A&& a) { return flatmap(f(std::forward<A>(a)), g); });
  }

  /** composition with sequential indexed iterations of rval_t-valued
   * function */
  template <typename G>
  auto
  and_then_loop(unsigned n, const G& g) const {

    return
      RvalM<A, giv_t<G>>::pure(
        [n, g, f=m_f](A&& a) {
          auto result = f(std::forward<A>(a));
          for (unsigned i = 0; i < n; ++i)
            result =
              flatmap(
                std::move(result),
                [i, g](auto&& r) {
                  return g(i, std::move(r));
                });
          return result;
        });
  }

  /** composition with sequential non-indexed iterations of rval_t-valued
   * function */
  template <typename G>
  auto
  and_then_repeat(unsigned n, const G& g) const {

    return
      and_then_loop(n, [g](unsigned, auto&& r) { return g(std::move(r)); });
  }

  /** composition with simple-valued function  */
  template <typename G>
  auto
  map(const G& g) const {

    return
      RvalM<A, std::invoke_result_t<G, B>>::pure(
        [g, f=m_f](A&& a) { return ::hpg::map(f(std::forward<A>(a)), g); });
  }
};

/** specialization of RvalMF<A> for A = void */
template <typename B, typename F>
struct RvalMF<void, B, F> :
  public RvalM<void, B> {

  static_assert(
    std::is_same_v<B, typename rval_value<std::invoke_result_t<F>>::type>);

  /** value type of G applied to value of type B */
  template <typename G>
  using gv_t = typename rval_value<std::invoke_result_t<G, B>>::type;

  /** value type of G applied to a loop index and value of type B */
  template <typename G>
  using giv_t = typename rval_value<std::invoke_result_t<G, unsigned, B>>::type;

  F m_f; /**< wrapped function, as value */

  /** constructor */
  RvalMF(const F& f)
    : m_f(f) {}

  /** constructor */
  RvalMF(F&& f)
    : m_f(std::move(f)) {}

  /** apply contained function to a value */
  rval_t<B>
  operator()() const {
    return m_f();
  }

  /** composition with rval_t-valued function */
  template <typename G>
  auto
  and_then(const G& g) const {

    return RvalM<void, gv_t<G>>::pure(
      [g, f=m_f]() { return flatmap(f(), g); });
  }

  /** composition with sequential iterations of rval_t-valued function */
  template <typename G>
  auto
  and_then_loop(unsigned n, const G& g) const {

    return
      RvalM<void, giv_t<G>>::pure(
        [n, g, f=m_f]() {
          auto result = f();
          for (unsigned i = 0; i < n; ++i)
            result =
              flatmap(
                std::move(result),
                [i, g](auto&& r) {
                  return g(i, std::move(r));
                });
          return result;
        });
  }

  /** composition with sequential non-indexed iterations of rval_t-valued
   * function */
  template <typename G>
  auto
  and_then_repeat(unsigned n, const G& g) const {

    return
      and_then_loop(n, [g](unsigned, auto&& r) { return g(std::move(r)); });
  }

  /** composition with simple-valued function  */
  template <typename G>
  auto
  map(const G& g) const {

    return RvalM<void, std::invoke_result_t<G, B>>::pure(
      [g, f=m_f]() { return ::hpg::map(f(), g); });
  }
};
#endif // HPG_API >= 17

}  // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
